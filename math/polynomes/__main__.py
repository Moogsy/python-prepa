from __future__ import annotations

from cmath import exp, polar, sqrt
from types import FunctionType
from itertools import dropwhile, zip_longest
from math import comb, factorial, inf, pi
from operator import not_
from typing import Literal, Iterable, Iterator, Protocol, TypeVar, Union

__all__ = ("Polynom",)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
Scalar = Union[float, complex]
j = exp(2j * pi / 3)


class PolynomError(Exception):
    ...


class RootNotFound(PolynomError):
    ...


class TooManyKnownRoots(PolynomError):
    ...


class NotARoot(PolynomError):
    ...


def hide_ones(x: Scalar) -> str:
    if x == 1:
        return ""
    return str(x)


def factor_negative(x: Scalar) -> tuple[Literal["+", "-"], Scalar]:
    if isinstance(x, complex):
        if max(x.imag, x.real) < 0:
            return ("-", -x)

        return ("+", x)

    if x < 0:
        return ("-", -x)

    return ("+", x)


class SupportsAdd(Protocol[T]):
    def __add__(self, __other: T) -> T:
        ...


class SupportsSub(Protocol[T]):
    def __sub__(self, __other: T) -> T:
        ...


class SupportsTrueDiv(Protocol[T]):
    def __truediv__(self, __other: T) -> T:
        ...


class SupportsMul(Protocol[T]):
    def __mul__(self, __other: T) -> T:
        ...


class SupportsPow(Protocol[T_co]):
    def __pow__(self, __other: int) -> T_co:
        ...

class SupportsCall(Protocol[T_co]): 
    def __call__(self, *__args, **kwargs) -> T_co: 
        ...

class ScalarLike(
    SupportsAdd[T], SupportsSub[T], SupportsMul[T], SupportsPow[T], Protocol[T]
):
    pass


class DivScalarLike(ScalarLike[T], SupportsTrueDiv[T], Protocol[T]):
    pass


class Polynom:
    __slots__ = ("_coefficients",)

    def __init__(self, coefficients: Iterable[Scalar], /):
        coeffs = tuple(coefficients)
        rev = dropwhile(not_, coeffs[::-1])
        self._coefficients = tuple(rev)[::-1]

    @classmethod
    def zero(cls):
        """Construct a polybom that constantly evalutates to 0"""
        return cls([])

    @classmethod
    def one(cls):
        """Constructs a polynom that constantly evaluates to 1"""
        return cls([1])

    @classmethod
    def constant(cls, a: Scalar, /):
        """
        Construct a polynom that constantly evaluates to "a"

        Args:
            a: constant that this polynom evaluates to
        """
        return cls([a])

    @classmethod
    def identity(cls):
        """Constructs a polynom that always returns the provided argument"""
        return cls([0, 1])

    @classmethod
    def Xpow(cls, n: int, /):
        """
        Constructs a polynom that returns the provided argument raised to the power n

        Args:
            n: power at which X is raised to
        """
        return cls([0 for _ in range(n)] + [1])

    @classmethod
    def from_roots(cls, roots_: Iterable[Scalar], /) -> Polynom:
        """
        Given a tuple (x_1, ..., x_n), constructs a polynom P 
        such as P(x_i) = 0 for all 1 <= i <= n

        Args:
            roots_: Iterable of points where this polynom should evaluate to 0
        """
        roots = tuple(roots_)
        prod = cls.one()

        for root in roots:
            prod *= Polynom([-root, 1])

        return prod

    @classmethod
    def lagrange(cls, points_: Iterable[tuple[Scalar, Scalar]], /) -> Polynom:
        """
        Given a set of points ((x_1, y_1), ... (x_n, y_n)), constructs a polynom 
        P such a P(x_i) = y_i for all 1 <= i <= n

        Args:
            points_: Iterable of points where the polynom's curve should go throught
        """
        points = tuple(points_)
        n = len(points)
        sum_ = cls.zero()

        for j in range(n):
            prod = cls([points[j][1]])
            for i in range(n):
                if i == j:
                    continue
                prod *= cls([-points[i][0], 1]) / (points[j][0] - points[i][0])

            sum_ += prod

        return sum_

    @classmethod
    def approximate_function(
        cls, f: SupportsCall, a: float, b: float, deg: int
    ) -> Polynom:
        """Approximates function "f" on [a, b] using a polynom of degree "deg"."""

        if deg == 0:
            return Polynom.constant(f((a + b) / 2))

        step = (b - a) / deg

        anchor_points = [a + k * step for k in range(deg + 1)]
        return cls.lagrange([(x, f(x)) for x in anchor_points])

    @classmethod
    def chebyshev(cls, kind: Literal[1, 2] = 1, n: int = 1) -> Polynom:
        """
        Chebyshev's polynoms

        Both kind are defined by the sequence:
            T_{n+2} = 2X T_{n+1} + T_n

        For the first kind:
            T_0 = 1
            T_1 = X

        For the second kind:
            T_0 = 1
            T_1 = 2X

        Args:
            kind: Which kind to choose
            n: Index of the required polynom
        """
        upper = int(n / 2)
        sum_ = cls.zero()

        if kind == 1:
            cst = cls([-1, 0, 1])

            for k in range(upper + 1):
                sum_ += comb(n, 2 * k) * (cst ** k) * cls.Xpow(n - 2 * k)

            return sum_

        if kind == 2:
            cst = cls([-1, 0, 1])

            for k in range(upper + 1):
                sum_ += comb(n + 1, 2 * k + 1) * (cst ** k) * cls.Xpow(n - 2 * k)

            return sum_

        raise TypeError(f"Expected 1 or 2 for parameter kind, got {kind}")

    @classmethod
    def hilbert(cls, n: int) -> Polynom:
        """
        Hilbert's polynoms

        Defined by:
            H_0 = 1

        for all non-zero natural integer n:
            H_n = X(X-1) ... (X - n + 1) / n!

        Args:
            n: Index of the desired polynom
        """
        prod_ = Polynom([1 / factorial(n)])

        for k in range(n):
            prod_ *= Polynom([-k, 1])

        return prod_

    @property
    def coefficients(self) -> tuple[Scalar, ...]:
        """
        Since P = a_0 + a_1 X + a_2 X^2 + ... + a_n X^n
        Returns the tuple (a_0, ..., a_n)
        """
        return self._coefficients

    @property
    def degree(self) -> int | float:
        """
        Returns this polynom's degree i.e. highest power.
        Note that if P(X) = 0, then it's degree is -infinity.
        """
        return len(self.coefficients) - 1 if self.coefficients else -inf

    @property
    def is_even(self) -> bool:
        """
        Checks whether this polynom is even
        """
        return all(a == 0 for a in self.coefficients[::2])

    @property
    def is_odd(self) -> bool:
        """
        Checks whether this polynom is odd
        """
        return all(a == 0 for a in self.coefficients[1::2])

    @property
    def is_real(self) -> bool:
        """
        Checks whether this polynom is real
        """
        return all(isinstance(a, float) for a in self.coefficients)

    @property
    def is_complex(self) -> bool:
        """
        Checks whether this polynom is complex
        """
        return any(isinstance(a, complex) for a in self.coefficients)

    def __repr__(self) -> str:
        return "{0.__class__.__name__}([{0.coefficients}])".format(self)

    def __str__(self) -> str:
        if self.degree < 0:
            return "0X^0"

        if self.degree == 0:
            return hide_ones(self.coefficients[0]) + "X^0"

        disp = []

        if self.coefficients[0] != 0:
            disp.append(str(self.coefficients[0]))

        if self.coefficients[1] != 0:
            sign, value = factor_negative(self[1])
            if value == 1:
                disp.append(f"{sign} X")
            else:
                disp.append(f"{sign} {value}X")

        for index, coeff in enumerate(self.coefficients[2:], start=2):
            if coeff == 0:
                continue

            sign, value = factor_negative(coeff)

            if value == 1:
                disp.append(f"{sign} X^{index}")
            else:
                disp.append(f"{sign} {value}X^{index}")

        return " ".join(disp)

    def __eq__(self, other: object) -> bool:
        return type(self) == type(other) and self.coefficients == other.coefficients  # type: ignore

    def __hash__(self) -> int:
        return hash(self.coefficients)

    def __bool__(self) -> bool:
        return self.degree >= 0

    def __len__(self) -> int | float:
        return self.degree

    def __iter__(self) -> Iterator[Scalar]:
        return iter(self.coefficients)

    def __call__(self, x: ScalarLike):
        sum_ = x - x
        for index, coeff in enumerate(self.coefficients):
            right = x ** index
            sum_ += coeff * right

        return sum_

    def __add__(self, other: Polynom) -> Polynom:
        coeffs = [
            x + y
            for x, y in zip_longest(self.coefficients, other.coefficients, fillvalue=0)
        ]
        return Polynom(coeffs)

    def __neg__(self) -> Polynom:
        return -1 * self

    def __sub__(self, other: Polynom) -> Polynom:
        return self + (-other)

    def __mul__(self, other: Scalar | Polynom) -> Polynom:

        if isinstance(other, (int, float, complex)):
            return Polynom([a * other for a in self.coefficients])

        # one of our polynomials is null
        if min(self.degree, other.degree) < 0:
            return Polynom.zero()

        coeffs = []
        for k in range(int(self.degree + other.degree) + 1):
            sum_ = 0
            for i in range(k + 1):
                try:
                    sum_ += self[i] * other[k - i]
                except IndexError:
                    continue

            coeffs.append(sum_)

        return Polynom(coeffs)

    __rmul__ = __mul__

    def __truediv__(self, other: Scalar) -> Polynom:
        return (1 / other) * self

    def __floordiv__(self, other: Polynom) -> Polynom:
        quotient, _ = self.__divmod__(other)
        return quotient

    def __mod__(self, other: Polynom) -> Polynom:
        _, remainder = self.__divmod__(other)
        return remainder

    def __matmul__(self, other: Polynom):
        return self(other)

    def __pos__(self):
        return self

    def __pow__(self, n: int):
        if n == 0:
            return Polynom.one()

        if self.degree == 0:
            return Polynom.constant(self[0] ** n)

        if self.degree == 1:
            b, a = self
            return Polynom([comb(n, k) * a**k * b**(n-k)  for k in range(n + 1)])

        p = self

        for _ in range(n - 1):
            p *= self

        return p

    def __getitem__(self, index: int):
        return self.coefficients[index]

    def __divmod__(self, other: Polynom) -> tuple[Polynom, Polynom]:
        if self.degree < other.degree:
            return Polynom.zero(), self

        A = self
        B = other
        Q = Polynom.zero()

        while A.degree >= B.degree:
            P = (A[-1] / B[-1]) * Polynom.Xpow(int(A.degree - B.degree))

            A -= P * B
            Q += P

        return Q, A

    def monic(self) -> Polynom:
        """
        Returns the monic polynomial associated to ours.
        """
        if self[-1] == 1:
            return self

        return self / self[-1]

    def derivative(self, n: int = 1) -> Polynom:
        """
        n-th derivative
        """
        if n > self.degree:
            return Polynom.zero()

        coeffs = list(self.coefficients)

        for _ in range(n):
            coeffs = [index * a for index, a in enumerate(coeffs)][1:]

        return Polynom(coeffs)

    def antiderivative(self, n: int = 1, constant: float | complex = 0, /):
        """
        n-th antiderivative using the provided constant
        """
        coeffs = list(self.coefficients)

        for _ in range(n):
            coeffs = [constant] + [a / index for index, a in enumerate(coeffs, start=1)]

        return Polynom(coeffs)

    def integrate(self, a: float, b: float, /) -> float:
        """
        Integral [a, b]
        """
        antiderivative = self.antiderivative()
        return antiderivative(b) - antiderivative(a)

    @classmethod
    def approximate_integral(
        cls, f: FunctionType, a: float, b: float, n: int, deg: int
    ) -> float:
        """
        Approximates the integral of "f" between "a" and "b" using "n" polynoms of degree "deg".
        """
        step = (b - a) / n
        integral = float(0)

        for k in range(n + 1):
            c = k * step
            d = (k + 1) * step
            P = cls.approximate_function(f, c, d, deg)
            integral += P.integrate(c, d)

        return integral

    def gcd(self, other: Polynom) -> Polynom:
        """
        Returns the greatest common divisor between this polynom and another one
        """
        P = self
        Q = other

        while Q:
            P, Q = Q, P % Q

        return P.monic()

    def lcm(self, other: Polynom) -> Polynom:
        """
        Returns the last common multiple between this polynom and another one
        """
        if not (self and other):
            return Polynom.zero()

        P = (self * other) // self.gcd(other)
        return P.monic()

    def _newton(self, x: Scalar, /) -> Scalar | None:
        derivative = self.derivative()

        for _ in range(10):
            try:
                x -= self(x) / derivative(x)
            except ZeroDivisionError:
                # Can't guarantee that it converged
                return None

        return x

    def _orig_newton(self, x: Scalar, /) -> Scalar:
        """
        Newton, but returns the original argument if it didn't converge
        """
        res = self._newton(x)
        if res is None:
            return x
        return res

    def _root_cubic(self, epsilon: float) -> set[Scalar]:
        """
        Finds the roots of this polynom if it's degree is 3.
        """
        d, c, b, a = self

        # yay
        delta = (
            18 * a * b * c * d
            - 4 * b ** 3 * d
            + (b * c) ** 2
            - 4 * a * c ** 3
            - 27 * (a * d) ** 2
        )
        delta_0 = b ** 2 - 3 * a * c
        delta_1 = 2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d

        if max(abs(delta), abs(delta_0)) <= epsilon:
            return {self._orig_newton(-b / (3 * a))}

        if abs(delta) <= epsilon and abs(delta_0) > epsilon:
            r_1 = self._orig_newton((9 * a * d - b * c) / (2 * delta_0))
            r_2 = self._orig_newton(
                (4 * a * b * c - 9 * a ** 2 * d - b ** 3) / (a * delta_0)
            )
            return {r_1, r_2}

        C = (delta_1 + sqrt(delta_1 ** 2 - 4 * delta_0 ** 3)) ** (1 / 3)

        close_roots = set()

        for k in range(3):
            jc = (j ** k) * C
            close_roots.add((-1 / (3 * a)) * (b + jc + delta_0 / jc))

        if (
            isinstance(delta, float) and delta > 0
        ):  # we know that all roots are real in that case
            close_roots = {z.real for z in close_roots}

        # Since it isn't as accurate when roots are bigger, we apply Newton's method
        return {self._orig_newton(root) for root in close_roots}

    def roots(
        self, known_roots: Iterable[Scalar] = (), epsilon: float = 10e-2
    ) -> set[Scalar] | bool:
        """
        If this polynom's degree is lower or equal to 3, returns all roots.

        Args:
            known_roots: Used to factor this polynom, pass atleast n - 3 of them
                         so the remaining ones can be found.
                         (n designates this polynom's degree)

            epsilon: Used to compute how close two floats needs to be
                     before they are considered equal.
                     Pass 0 to avoid approximating anything.

        Returns:
            True: this polynom is null, everything is a root

            None: No roots

            set(...): A set containing all of this polynom's roots

        Raises:
            NotARoot: One element provided in known_roots is not a root
                      (Or not doesn't evaluates close enough to 0 according
                      to the provided epsilon)

            TooManyKnownRoots: Too many provided known_roots, can't have more than
                               this polynom's degree

            RootNotFound: Couldn't find remaining roots because not enough known_roots
                          were provided

        Note:
            Apparently, numpy has a method to find the roots of any polynom
            Will switch to it if I ever understand how it works
        """
        known_roots = set(known_roots)
        for maybe_root in known_roots:
            if abs(self(maybe_root)) > epsilon:
                raise NotARoot(f"{maybe_root} is not a root of this polynom")

        if self.degree < 0:  # P(X) = 0
            return True

        if self.degree == 0:  # a = 0
            return set()

        if self.degree == 1:  # az + b = 0
            return {-self[0] / self[1]}

        if self.degree == 2:  # c + bz + az^2 = 0
            c, b, a = self
            s_delta = sqrt(b ** 2 - 4 * a * c)
            return {(-b - s_delta) / (2 * a), (-b + s_delta) / (2 * a)}

        if set(self.coefficients[1:-1]) == {0}:  # z**n + a = 0
            if self[0] == 0:  # z**n = 0
                return {0}

            r, phi = polar(-self[0])
            n = int(self.degree)

            return {r ** (1 / n) * exp((1j / n) * (phi + 2 * k * pi)) for k in range(n)}

        if self.degree == 3:  # az^3 + bz^2 + cz + d = 0
            return self._root_cubic(epsilon)

        diff = int(self.degree) - len(known_roots)

        if diff < 0:
            raise TooManyKnownRoots("Too many known roots provided")

        if diff == 0:
            return known_roots

        # Try some basic stuff to get more roots
        guessed_roots: set[complex] = set()

        for known_root in known_roots:
            if self.is_real:  
                # if we have root, then it's conjugate is a root too
                conj = known_root.conjugate()

                if abs(known_root - conj) > epsilon:
                    guessed_roots.add(conj)

                neg = -known_root
                if self.is_even or self.is_odd:
                    # if we have a root, then it's opposite is a root too
                    if abs(known_root - neg) > epsilon:
                        guessed_roots.add(neg)

        known_roots |= guessed_roots

        if diff <= 3:  # ensure that the new polynom has roots
            poly = self // Polynom.from_roots(known_roots)
            new_roots = poly.roots()

            if isinstance(new_roots, set):
                if not any(isinstance(x, bool) for x in new_roots):  # type checker ...
                    return known_roots | new_roots

        raise RootNotFound("No roots found for this polynom")
