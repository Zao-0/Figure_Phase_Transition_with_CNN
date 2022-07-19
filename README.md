# Figure_Phase_Transition_with_CNN
This is my final project of a course. Detailed info is in README.

## Background info
We have Heisenberg model with Hamiltonian:
$$H=-J\sum_{i}^{L}(\vec \sigma_i \vec\sigma_{i+1})-\sum_{i}^{L} h \sigma_i^z $$
With the Hamiltonian, we could get eigenstates with corresponding eigenenergy close to 0. Use these eigenstates to create density matrix. Then compute the density matrix for n adjacent spins. Training set: $h\in[-0.5J,0.5J]$ with label 0(localized), $h\in[-8J,8J]$ with label 1(extended). Other h's inervals are used as test set.
