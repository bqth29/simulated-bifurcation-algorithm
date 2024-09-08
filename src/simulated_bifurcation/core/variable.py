import re

from .optimization_domain import OptimizationDomain


class Variable:
    INTEGER_REGEX = re.compile("^int[1-9][0-9]*$")
    DOMAIN_ERROR = ValueError(
        'Domain type must be one of "spin" or "binary", or be a string starting '
        'with "int" and followed by a positive integer that represents '
        "the number of bits required to encode the values of the domain. "
        "More formally, it should match the following regular expression: "
        '"^int[1-9][0-9]*$" (ex: "int7", "int42", ...).'
    )

    def __init__(self, domain_type: OptimizationDomain, encoding_bits: int) -> None:
        if (
            domain_type == OptimizationDomain.SPIN
            or domain_type == OptimizationDomain.BINARY
        ) and encoding_bits != 1:
            raise ValueError(
                "A spin or binary variable can only be encoded with one bit."
            )
        self.__domain_type = domain_type
        self.__encoding_bits = encoding_bits

    @property
    def encoding_bits(self) -> int:
        """
        Number of bits required to encode the values of the variable.
        """
        return self.__encoding_bits

    @property
    def is_spin(self) -> int:
        """
        Returns `True` is the variable is a spin variable, `False` otherwise.
        """
        return self.__domain_type == OptimizationDomain.SPIN

    @staticmethod
    def from_str(domain: str):
        """
        Instantiates a Variable from a domain type name.

        Parameters
        ----------
        domain : str
            Domain type. Must be `"spin"`, `"binary"` of match the regular expression
            `"^int[1-9][0-9]*$"`. For integers, the number following `"int"` corresponds
            to the number of bits required to encode the variable possible values.

        Returns
        -------
        Variable
            Variable defined on the input domain.

        Raises
        ------
        ValueError
            If the domain is not a valid one (spin, binary or integer).
        """
        if domain == "spin":
            return Variable(OptimizationDomain.SPIN, 1)
        if domain == "binary":
            return Variable(OptimizationDomain.BINARY, 1)
        if Variable.INTEGER_REGEX.match(domain) is None:
            raise Variable.DOMAIN_ERROR
        return Variable(OptimizationDomain.INTEGER, int(domain[3:]))
