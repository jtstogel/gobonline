#include "src/util/union_find.h"

#include "gtest/gtest.h"

namespace gobonline {

TEST(UnionFind, UnitarySets) {
  UnionFind uf(10);
  EXPECT_EQ(uf.Sets().size(), 10);
}

TEST(UnionFind, Union) {
  UnionFind uf(2);
  EXPECT_NE(uf.FindSet(0), uf.FindSet(1));
  uf.Union(0, 1);
  EXPECT_EQ(uf.FindSet(0), uf.FindSet(1));
}

TEST(UnionFind, TransitiveSetMembership) {
  UnionFind uf(1000);
  for (size_t i = 0; i < uf.Size() - 1; i++) {
    uf.Union(i, i + 1);
  }
  EXPECT_EQ(uf.FindSet(0), uf.FindSet(999));
}

}  // namespace gobonline
