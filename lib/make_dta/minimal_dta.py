from typing import Any


class MinimalDTA:

    def __init__(
            self,
            drug_ids: dict[int, Any],
            target_ids: dict[int, Any],
            known_affinities: dict[tuple[int, int]]
    ):
        self.drug_ids = drug_ids
        self.target_ids = target_ids
        self.known_true = set()
        self.known_false = set()
        for index, value in known_affinities.items():
            if value:
                self.known_true.add(value)
            else:
                self.known_false.add(value)

    def __getitem__(self, item) -> bool:
        return item in self.known_true
