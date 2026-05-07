from itertools import combinations

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank= {x: 0 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self):
        groups = {}
        for x in self.parent:
            groups.setdefault(self.find(x), []).append(x)
        return groups


def transitivity_violations(models, same_cluster):
    def get(a, b):
        if (a, b) in same_cluster:
            return same_cluster[(a, b)]
        return same_cluster[(b, a)]

    def has(a, b):
        return (a, b) in same_cluster or (b, a) in same_cluster

    n_antecedents = 0
    violations = []

    for a, b, c in combinations(models, 3):
        if not (has(a, b) and has(b, c) and has(a, c)):
            continue
        ab, bc, ac = get(a, b), get(b, c), get(a, c)
        for x, y, z, xy, yz, xz in [
            (a, b, c, ab, bc, ac),
            (a, c, b, ac, bc, ab),
            (b, c, a, bc, ac, ab),
        ]:
            if xy == 1 and yz == 1:
                n_antecedents += 1
                if xz != 1:
                    violations.append((x, y, z))

    return n_antecedents, len(violations), violations


def cluster_density_metrics(models, same_cluster):
    n = len(models)
    pairs = list(combinations(models, 2))
    n_pairs = len(pairs)

    known = [(a, b) for a, b in pairs if (a, b) in same_cluster or (b, a) in same_cluster]

    def label(a, b):
        return same_cluster.get((a, b), same_cluster.get((b, a)))

    positives = [(a, b) for a, b in known if label(a, b) == 1]
    pos_rate = (len(positives) / len(known)) if known else 0.0

    uf = UnionFind(models)
    for a, b in positives:
        uf.union(a, b)
    comps = uf.components()

    sizes = [len(v) for v in comps.values()]
    largest = max(sizes) if sizes else 0

    intra_values = []
    intra_known_total = 0
    intra_pos_total = 0
    for members in comps.values():
        if len(members) < 2:
            continue
        comp_pairs = list(combinations(members, 2))
        comp_known = [
            (a, b) for a, b in comp_pairs
            if (a, b) in same_cluster or (b, a) in same_cluster
        ]
        if not comp_known:
            continue
        comp_pos = sum(1 for a, b in comp_known if label(a, b) == 1)
        intra_known_total += len(comp_known)
        intra_pos_total += comp_pos
        intra_values.append(comp_pos / len(comp_known))

    intra_density = (intra_pos_total / intra_known_total) if intra_known_total else 0.0

    return {
        "n_models": n,
        "n_pairs": n_pairs,
        "n_known_pairs": len(known),
        "positive_rate": pos_rate,
        "n_components": len(comps),
        "largest_component_size": largest,
        "largest_component_frac": (largest / n) if n else 0.0,
        "intra_cluster_density": intra_density,
        "modularity_like": intra_density - pos_rate,
        "n_singletons": sum(1 for s in sizes if s == 1),
    }