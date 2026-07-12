# Network Theorems and Analysis

A **network** is a combination of components, such as resistances and voltage sources, interconnected to achieve a particular end result. The network theorems usually provide shorter methods for solving a circuit.

## Superposition Therorem

The **superposition theorem** is a principle for analyzing _linear circuits_ with _multiple sources_, which states that "the _total current or voltage_ for any element is the algebraic sum of the effects of each source acting separately."

> A **linear circuit** is an electrical circuit where the relationship between voltage and current is directly proportional, meaning an increase in input causes a proportional increase in output.

> Voltage divider: $V_{\text{parallel}} = V_{\text{source}} \frac{R_{\text{parallel}}}{R_{\text{series}} + R_{\text{parallel}}}$

> Negative voltage is valid and still provides charge. Because voltage is about the potential energy and direction. For example, the voltage from point $A$ with $5V$ and point $B$ at $0V$ is $V_{AB} = V_A - V_B = 5V - 0V = 5V$. But from point $B$ to point $A$ is $V_{BA} = V_B - V_A = 0V - 5V = -5V$

> Disable voltage source by short circuiting it (connecting the circuit it to ground instead)

## Thevenin's Theorem

This theorem is very useful in _simplifying the process of solving_ for the _unknown values of voltage and current_ in a network.

## Norton's Theorem

This theorem is used to simplify a network in terms of currents instead of voltages.

Any network in the block at the left can be reduced to the Norton equivalent parallel circuit at the right.

The idea of current source is that it supplies a total line current to be divided among parallel branches, corresponding to a voltage source applying a total voltage to be divided among series components.