# translated_imo_2002_p2a
## Formal goal
- `eqangle e c e j e j e f`
## Proof plan
1. Parse the construction into 7 clauses over 14 symbolic points.
2. Ground constructor semantics for `midpoint` to expose implicit incidences, equalities, and perpendicular/parallel relations.
3. Goal-directed strategy: transform the target into cyclicity or parallelism to obtain angle equality.
4. Highest-scoring reusable lemmas/rules:
   1. [score=3] `eqangle a b c d m n p q, eqangle c d e f p q r u => eqangle a b e f m n r u`
   2. [score=2] `cyclic A B P Q => eqangle P A P B Q A Q B`
   3. [score=2] `perp A B C D, perp E F G H, npara A B E F => eqangle A B E F C D G H`
   4. [score=2] `cong O A O B, ncoll O A B => eqangle O A A B A B O B`
   5. [score=2] `circle O A B C, perp O A A X => eqangle A X A B C A C B`
5. Attempt a machine-verifiable derivation by instantiating the retrieved rules on symbols introduced in the construction; unresolved variable bindings are left explicit for downstream search.
