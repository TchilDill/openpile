from openpile.construct import Pile


def txt_pile(obj: Pile) -> str:
    """Function that creates several lines of text with the pile data.

    Parameters
    ----------
    obj : openpile.construct.Pile
        Pile instance

    Returns
    -------
    str
        output text with pile data
    """

    # Create string to document pile
    txt = "{:-^80s}".format("")
    txt += "\n{:^80s}".format("Pile Input")
    txt += "\n{:-^80s}".format("")
    txt += f"\nPile Material: {obj.material.name}"
    txt += f"\tPile Type: {obj.type}"
    txt += f"\tYoung modulus: {obj._young_modulus/1000:.0f} MPa"
    txt += f"\tPoisson ratio = {obj.material.poisson:.2f}"
    txt += f"\nMaterial Unit Weight: {obj.material.unitweight:0.1f} kN/m3"
    txt += f"\n\nPile sections:\n"
    txt += obj.data.to_string(header=True, index=True)

    return txt
