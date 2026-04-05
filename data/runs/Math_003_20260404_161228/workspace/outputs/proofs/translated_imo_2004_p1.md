# translated_imo_2004_p1
## Formal goal
- `coll p b c`
## Proof plan
1. Parse the construction into 8 clauses over 16 symbolic points.
2. Ground constructor semantics for `angle_bisector, circle, midpoint` to expose implicit incidences, equalities, and perpendicular/parallel relations.
3. Goal-directed strategy: show two candidate points lie on a common line by chaining line-incidence constructions and parallel/perpendicular lemmas.
4. Highest-scoring reusable lemmas/rules:
   1. [score=1] `circle O A B C, perp O A A X => eqangle A X A B C A C B`
   2. [score=1] `circle O A B C, eqangle A X A B C A C B => perp O A A X`
   3. [score=1] `circle O A B C, midp M B C => eqangle A B A C O B O M`
   4. [score=1] `circle O A B C, coll M B C, eqangle A B A C O B O M => midp M B C`
   5. [score=1] `circle O A B C, coll O A C => perp A B B C`
5. Attempt a machine-verifiable derivation by instantiating the retrieved rules on symbols introduced in the construction; unresolved variable bindings are left explicit for downstream search.
