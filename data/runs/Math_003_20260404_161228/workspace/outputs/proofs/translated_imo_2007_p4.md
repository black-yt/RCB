# translated_imo_2007_p4
## Formal goal
- `eqratio k k1 l l1 r q r p`
## Proof plan
1. Parse the construction into 9 clauses over 18 symbolic points.
2. Ground constructor semantics for `circle, foot, midpoint` to expose implicit incidences, equalities, and perpendicular/parallel relations.
3. Goal-directed strategy: seek similar triangles or midpoint/parallel configurations to prove equal ratios.
4. Highest-scoring reusable lemmas/rules:
   1. [score=3] `eqratio a b c d m n p q, eqratio c d e f p q r u => eqratio a b e f m n r u`
   2. [score=2] `midp M A B, midp N C D => eqratio M A A B N C C D`
   3. [score=1] `circle O A B C, perp O A A X => eqangle A X A B C A C B`
   4. [score=1] `circle O A B C, eqangle A X A B C A C B => perp O A A X`
   5. [score=1] `circle O A B C, midp M B C => eqangle A B A C O B O M`
5. Attempt a machine-verifiable derivation by instantiating the retrieved rules on symbols introduced in the construction; unresolved variable bindings are left explicit for downstream search.
