# translated_imo_2003_p4
## Formal goal
- `cong p q q r`
## Proof plan
1. Parse the construction into 9 clauses over 17 symbolic points.
2. Ground constructor semantics for `circle, foot` to expose implicit incidences, equalities, and perpendicular/parallel relations.
3. Goal-directed strategy: reduce the target to equal distances and search for circles, midpoints, or reflections that imply radius equalities.
4. Highest-scoring reusable lemmas/rules:
   1. [score=3] `eqratio A B P Q C D U V, cong P Q U V => cong A B C D`
   2. [score=2] `cyclic A B C P Q R, eqangle C A C B R P R Q => cong A B P Q`
   3. [score=2] `eqangle6 A O A B B A B O, ncoll O A B => cong O A O B`
   4. [score=2] `perp A B B C, midp M A C => cong A M B M`
   5. [score=2] `midp M A B, perp O M A B => cong O A O B`
5. Attempt a machine-verifiable derivation by instantiating the retrieved rules on symbols introduced in the construction; unresolved variable bindings are left explicit for downstream search.
