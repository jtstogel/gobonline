#include "AccelStepper.h"
#include "Arduino.h"
#include "MultiStepper.h"

#define MAX_SPEED 500
#define ACCELERATION 100

AccelStepper stepper1(AccelStepper::FULL2WIRE, /*dir=*/2, /*step=*/3);
bool stepper1_running = false;

AccelStepper stepper2(AccelStepper::FULL2WIRE, /*dir=*/8, /*step=*/9);
bool stepper2_running = false;

constexpr int kNumSteppers = 2;
AccelStepper* steppers[kNumSteppers];
MultiStepper steppers_controller;

constexpr int kIdLen = 4;

void setup() {
  Serial.begin(9600);

  steppers[0] = &stepper1;
  steppers[1] = &stepper2;

  for (int i = 0; i < kNumSteppers; i++) {
    steppers[i]->setCurrentPosition(0);
    steppers[i]->setMaxSpeed(MAX_SPEED);
    steppers_controller.addStepper(*steppers[i]);
  }
}

struct TextPointer {
  char* s;
  int length;
  int idx;
};

/**
 * Matches text against a null-terminated prefix string.
 *
 * If it is a match, `idx` is incremented to point to the character after
 * the provided prefix.
 */
bool SkipPrefix(const char* prefix, TextPointer* p) {
  for (int i = 0; (i + p->idx) < p->length; i++) {
    if (prefix[i] == '\0') {
      p->idx += i;
      return true;
    }

    if (p->s[i + p->idx] != prefix[i]) {
      return false;
    }
  }

  return false;
}

/**
 * Consumes `len` characters of `p`.
 *
 * Returns false if there are not enough characters.
 */
bool ConsumeFixedLength(char* dst, int len, TextPointer* p) {
  if (p->idx + len >= p->length) {
    return false;
  }
  memcpy(dst, &p->s[p->idx], len);
  p->idx += len;
  return true;
}

bool IsNumericCharacter(char c) { return '0' <= c && c <= '9'; }

/**
 * Consumes the next integer in `p`.
 *
 * Returns false if the next character is not a number.
 */
bool ConsumeInteger(int32_t* dst, TextPointer* p) {
  if (p->idx >= p->length) {
    return false;
  }

  if (!IsNumericCharacter(p->s[p->idx])) {
    return false;
  }

  int32_t result = 0;

  static constexpr int kMaxDecimalLen = 9;
  int len = 0;

  int idx = p->idx;
  while (idx < p->length && IsNumericCharacter(p->s[idx])) {
    result *= 10;
    result += static_cast<int32_t>(p->s[idx] - '0');
    idx++;

    if (++len > kMaxDecimalLen) {
      return false;
    }
  }

  p->idx = idx;
  *dst = result;

  return true;
}

void GetPositions(int32_t positions[kNumSteppers]) {
  for (int i = 0; i < kNumSteppers; i++) {
    positions[i] = steppers[i]->currentPosition();
  }
}

void SetPositions(int32_t positions[kNumSteppers]) {
  steppers_controller.moveTo(positions);
  steppers_controller.runSpeedToPosition();
}

/**
 * Commands are encoded as strings and passed along the wire as text.
 *
 * All commands are prefixed with a unique ID that is 8 lowercase alpha chars.
 * All commands end with a newline character.
 *
 * PING
 * A ping is just a check from the client to make sure this device exists
 * and is reponding. The request and response is just a string as below,
 * where {id} is some client-provided text less than 16 characters in length.
 *   request:  cmd:{id}:ping
 *   response: res:{id}:pong
 *
 * SET_POSITIONS
 * The SET_POSITIONS command instructs the microcontroller to move all the
 * steppers to some positions. A position is a positive integer.
 * This command blocks until the steppers move to their positions.
 *   request:  cmd:{id}:set_positions:{pos},{pos}
 *   response: res:{id}:set_positions
 *
 * GET_POSITIONS
 * The GET_POSITIONS command reads the positions of all stepper motors and
 * returns those values to the client. A position is a positive integer.
 *   request:  cmd:{id}:get_positions
 *   response: res:{id}:get_positions:{pos},{pos}
 */
void DoCommand() {
  char cmd[128];
  int len = Serial.readBytesUntil('\n', cmd, sizeof(cmd) - 1);
  if (len <= 0) {
    return;
  }
  cmd[len] = '\0';

  TextPointer p = {
      .s = cmd,
      .length = len,
      .idx = 0,
  };

  if (!SkipPrefix("cmd:", &p)) return;

  char id[kIdLen + 1];
  id[kIdLen] = '\0';
  if (!ConsumeFixedLength(id, kIdLen, &p)) return;

  if (!SkipPrefix(":", &p)) return;

  char res[128];
  if (SkipPrefix("ping", &p)) {
    snprintf(res, sizeof(res), "res:%s:pong\n", id);
    Serial.write(res);
    return;
  }

  if (SkipPrefix("set_positions:", &p)) {
    int32_t positions[kNumSteppers];
    static_assert(kNumSteppers == 2, "expected kNumSteppers == 2");
    if (!ConsumeInteger(&positions[0], &p)) return;
    if (!SkipPrefix(",", &p)) return;
    if (!ConsumeInteger(&positions[1], &p)) return;

    SetPositions(positions);

    snprintf(res, sizeof(res), "res:%s:set_positions\n", id);
    Serial.write(res);

    return;
  }

  if (SkipPrefix("get_positions", &p)) {
    int32_t positions[2];
    GetPositions(positions);

    snprintf(res, sizeof(res), "res:%s:get_positions:%ld,%ld\n", id,
             positions[0], positions[1]);
    Serial.write(res);
    return;
  }

  return;
}

void loop() {
  DoCommand();
  delay(10);
}
