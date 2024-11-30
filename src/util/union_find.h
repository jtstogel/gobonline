#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gobonline {

/**
 * A basic, unoptimized Union Find.
 */
class UnionFind {
 public:
  explicit UnionFind(size_t n);

  /** Combines the sets containing the two elements. */
  void Union(size_t el1, size_t el2);

  /** Returns the label of the set containing `el`. */
  size_t FindSet(size_t el);

  /** Returns the number of elements in this union find. */
  size_t Size();

  /** Returns the sets in this union find. */
  std::vector<std::vector<size_t>> Sets();

 private:
  size_t FindRoot(size_t element_idx);

  struct Node {
    uint32_t parent_idx;
  };

  std::vector<Node> nodes_;
};

}  // namespace gobonline
