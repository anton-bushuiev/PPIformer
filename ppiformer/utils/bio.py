from Bio.PDB.Polypeptide import protein_letters_3to1
from graphein.protein.resi_atoms import BASE_AMINO_ACIDS


BASE_AMINO_ACIDS = BASE_AMINO_ACIDS
BASE_AMINO_ACIDS_INVERSE = {aa: i for i, aa in enumerate(BASE_AMINO_ACIDS)}
BASE_AMINO_ACIDS_GROUPED = [
    'R', 'H', 'K',
    'D', 'E',
    'S', 'T', 'N', 'Q',
    'C', 'G', 'P',
    'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'
]

def amino_acid_to_class(aa: str) -> int:
    aa = aa.upper()
    if len(aa) == 3:
        aa = protein_letters_3to1[aa]
    return BASE_AMINO_ACIDS_INVERSE[aa]


def class_to_amino_acid(c: int) -> str:
    return BASE_AMINO_ACIDS[c]
