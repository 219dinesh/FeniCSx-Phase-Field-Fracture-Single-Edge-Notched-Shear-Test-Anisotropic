# Phase-Field Fracture Simulation in FEniCSx (Shear Test)

## Overview
This repository contains a Python script for simulating quasi-static brittle fracture using the **Phase-Field approach** in FEniCSx. The code models a 2D unit square domain with a pre-existing horizontal notch, subjected to pure shear loading on the top boundary. 

To prevent unrealistic crack propagation under compression, the model employs a **volumetric-deviatoric strain energy split** (anisotropic damage model), ensuring that damage only accumulates under tensile or shear strains. The coupled system is solved using a staggered (alternate minimization) scheme with an irreversible history field.

## Dependencies
To run this code, you need an environment with the FEniCSx (dolfinx) suite installed. 
* **dolfinx** (Tested with v0.6.0+)
* **ufl**
* **petsc4py** & **PETSc**
* **mpi4py**
* **numpy**
* **ParaView** (for viewing the `.xdmf` output files)

## Problem Description & Geometry
* **Domain:** A 2D Unit Square ($L = 1.0 \times 1.0$).
* **Mesh:** A uniform structured quadrilateral/triangular mesh with element size $h = 0.004$ (250 x 250 elements), ensuring the length scale parameter ($l_0 = 0.015$) is sufficiently resolved ($l_0 \approx 4h$).
* **Initial Flaw:** A horizontal pre-crack is explicitly defined from $x = 0$ to $x = 0.5$ at mid-height ($y = 0.5$). The initial crack is seeded using an exponential damage profile and enforced geometrically via a Dirichlet boundary condition.

### Boundary Conditions
The model simulates a pure shear test:
* **Bottom Edge ($y=0$):** Fully clamped ($u_x = 0, u_y = 0$).
* **Top Edge ($y=1$):** Prescribed horizontal displacement increasing linearly with time ($u_x = \Delta u \cdot t, u_y = 0$).
* **Left & Right Edges ($x=0, x=1$):** Constrained against vertical displacement ($u_y = 0$), allowing horizontal sliding.

## Material Parameters
The code uses the following material properties:

| Parameter | Symbol | Value | Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Young's Modulus** | $E$ | $210,000.0$ | MPa | Defines the linear elastic stiffness. |
| **Poisson's Ratio** | $\nu$ | $0.2$ | - | Ratio of transverse to axial strain. |
| **Fracture Toughness** | $G_c$ | $2.7$ | N/mm | Critical energy release rate for crack propagation. |
| **Length Scale** | $l_0$ | $0.015$ | mm | Regularization parameter dictating the crack band width. |
| **Residual Stiffness** | $k_{res}$ | $1 \times 10^{-6}$ | - | Small numerical parameter to prevent matrix singularity when fully cracked. |

## Governing Equations

### 1. Kinematics and Strain Energy Split
The infinitesimal strain tensor is defined as:
$$\boldsymbol{\varepsilon}(\mathbf{u}) = \frac{1}{2} \left[ \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right]$$

To model anisotropic damage (cracks only grow under tension/shear), the strain is decomposed into a deviatoric part ($\varepsilon_{dev}$) and a volumetric trace. The elastic strain energy density is split into active ($\psi^+$) and passive ($\psi^-$) components:

$$\psi^+ = \frac{1}{2} K_0 \langle \mathrm{tr}(\varepsilon) \rangle_+^2 + \mu (\varepsilon_{dev} : \varepsilon_{dev})$$

$$\psi^- = \frac{1}{2} K_0 \langle \mathrm{tr}(\varepsilon) \rangle_-^2$$

Where $K_0 = \lambda + \mu$ is the effective bulk modulus for plane assumptions, $\mu$ is the shear modulus, and $\langle \cdot \rangle_{\pm}$ denotes the Macaulay brackets.

### 2. Displacement Formulation (Linear Momentum)
Damage degrades only the active part of the strain energy. The degradation function follows the standard AT2 formulation: $g(d) = (1-d)^2 + k_{res}$. The Cauchy stress tensor is:

$$\sigma = [(1-d)^2 + k_{res}] \left( K_0 \langle \mathrm{tr}(\varepsilon) \rangle_+ \mathbf{I} + 2\mu \varepsilon_{dev} \right) + K_0 \langle \mathrm{tr}(\varepsilon) \rangle_- \mathbf{I}$$

The weak form solved for the displacement $\mathbf{u}$ with test function $\mathbf{v}$ is:
$$\int_{\Omega} \boldsymbol{\sigma} : \boldsymbol{\varepsilon}(\mathbf{v}) \, d\mathbf{x} = 0$$

### 3. Damage Formulation (Phase-Field)
The phase-field variable $d \in [0,1]$ distinguishes the intact material ($d=0$) from the fully broken material ($d=1$). The evolution is driven by the history field $\mathcal{H}$, which satisfies the irreversibility condition ($\dot{d} \geq 0$) by capturing the maximum active strain energy reached throughout the loading history:
$$\mathcal{H}(\mathbf{x}, t) = \max_{\tau \in [0, t]} \psi^+(\mathbf{x}, \tau)$$

The weak form solved for damage $d$ with test function $w$ is:
$$\int_{\Omega} \left[ \frac{G_c}{l_0} d \cdot w + G_c l_0 \nabla d \cdot \nabla w \right] d\mathbf{x} = \int_{\Omega} 2(1-d) \mathcal{H} w \, d\mathbf{x}$$

## Code Structure
1. **Geometry & Mesh:** Creates a $250 \times 250$ uniform quadrilateral mesh over a unit square using MPI.
2. **Pre-Crack Initialization:** An exponential damage profile $d(x) = \exp(-x/l_0)$ is mapped to the nodes near the mid-plane to physically seed the notch. The initial history field is synced to match this damage.
3. **Function Spaces:** Defines a Vector space `V_u` for displacements and a Scalar space `V_d` for the phase-field.
4. **Boundary Conditions:** Defines geometric locators for the edges and applies Dirichlet BCs. It also marks the top boundary to compute total reaction forces later.
5. **Variational Forms (`ufl`):** Translates the governing equations (stress, strain, degradation, phase-field evolution) into weak forms.
6. **Nonlinear Solvers:** Configures PETSc SNES solvers using a direct `mumps` solver for robust convergence.
7. **Time-Stepping Loop:** Increments the boundary displacement ($\Delta u = 0.0001$), solves for $\mathbf{u}$, updates the history field $\mathcal{H}$, solves for $d$, computes reaction forces, and writes outputs to disk.

## Output & Visualization
The script generates two `.xdmf` files in the specified directory:
* `Displacement_Out_line_shear_anisotropic_amor_try.xdmf`
* `Damage_Out_line_shear_anisotropic_amor_try.xdmf`

**To view the results:**
1. Open ParaView.
2. Load the `.xdmf` files.
3. Select the `Xdmf3ReaderT` when prompted.
4. Apply the **Warp By Vector** filter to the displacement file to visualize the structural deformation.
5. Overlay the damage field to watch the crack propagate from the notch to the boundaries under shear forces.
