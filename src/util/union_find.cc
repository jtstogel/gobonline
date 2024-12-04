#include "src/util/union_find.h"

#include <map>
#include <vector>

namespace gobonline {

UnionFind::UnionFind(size_t n) : nodes_(n) {
  for (size_t i = 0; i < n; i++) {
    nodes_[i].parent_idx = i;
  }
}

size_t UnionFind::FindRoot(size_t element_idx) {
  size_t node_idx = element_idx;
  size_t parent_idx = nodes_[node_idx].parent_idx;
  while (node_idx != parent_idx) {
    // New parent is the node's grandparent.
    // This is a lazy method of path compression.
    nodes_[node_idx].parent_idx = nodes_[parent_idx].parent_idx;

    node_idx = parent_idx;
    parent_idx = nodes_[parent_idx].parent_idx;
  }
  return node_idx;
}

void UnionFind::Union(size_t el1, size_t el2) { nodes_[el2].parent_idx = el1; }

size_t UnionFind::FindSet(size_t el) { return FindRoot(el); }

size_t UnionFind::Size() { return nodes_.size(); }

std::vector<std::vector<size_t>> UnionFind::Sets() {
  std::map<size_t, std::vector<size_t>> sets_by_label;
  for (size_t el = 0; el < nodes_.size(); el++) {
    auto it = sets_by_label.insert({FindSet(el), {}}).first;
    it->second.push_back(el);
  }

  std::vector<std::vector<size_t>> sets;
  sets.reserve(sets_by_label.size());
  for (auto& entry : sets_by_label) {
    sets.push_back(std::move(entry.second));
  }
  return sets;
}

}  // namespace gobonline
