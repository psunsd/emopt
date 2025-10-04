"""
/******************************************************************************
 * Copyright (c) 2023, Andrew Michaels.  All rights reserved.
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

This modules provides the definition for the :class:`.AdjointMethod` class.  Given an
electromagnetic structure (which is simulated using the FDFD class) and a merit
function which describes the 'performance' of that electromagnetic structure,
the :class:`.AdjointMethod` class defines the methods needed in order to calculate the gradient
of that merit function with respect to a set of user-defined design variables.

Notes
-----

Mathematically, the adjoint method calculates the gradient of a function
:math:`F(\mathbf{E}, \mathbf{H})` which has an explicit dependence on the
electric and magnetic fields (:math:`\mathbf{E}` and :math:`\mathbf{H}`).
Assuming we have expressed Maxwell's equations as a discretized linear system
of equations, one can show [1]_ that the derivatives of :math:`F` are given by

.. math::
    \\frac{d F}{d p_i} = -2\\Re\\left\{ y^T \\frac{\\partial A}{\\partial p_i} x\\right\\}

where :math:`x` contains the electric and magnetic fields, :math:`y`
contains a second set of 'adjoint' fields which are found by solving a second
set of linear system of equations which consist of the transposed Maxwell's
equations, and :math:`\partial A / \partial p_i` describes how the materials in
the system change with respect to changes to the design variables of the
system.

The AdjointMethod class does most of the work needed to compute :math:`x`,
:math:`y`, :math:`\partial A / \partial p_i`, and the gradient
:math:`\\nabla_\\mathbf{p}F`.

More generally, we may specify a function of which depndends not only on the
fields, but also explicitly on the design variables. In this case, the function
is given by

.. math::
    F \\rightarrow F(\\mathbf{E}, \\mathbf{H}, p_1, p_2, \cdots, p_N)

The derivative of this function with respect to the i'th
design variable, :math:`p_i` is given by

.. math::
    \\frac{d F}{d p_i} = -2\\Re\\left\{ y^T
    \\frac{\\partial A}{\\partial p_i} x\\right\\} + \\frac{\\partial
    F}{\\partial p_i}

The derivative with respect to :math:`p_i` on the right-hand side is assumed to
be known, thus very general figures of merit can be computed using the adjoint
method.

Note: This file uses MPI for parallelism.  As a result, return types and values
will depend on the RANK of the node running the code.

Examples
--------

The AdjointMethod class is used by extending the AdjointMethod base class. At
a minimum, four methods must be defined in the inheriting class. As an
example, a custom AdjointMethod class might look like

.. doctest::

    class MyAdjointMethod(AdjointMethod):

        def __init__(self, sim, myparam, step=1e-8):
            super(MyAdjointMethod, self).__init__(sim, step=step)
            self._myparam = myparam

        def update_system(self, params):
            # update system based on values in params
            ...

        def calc_fom(self, sim, params):
            # calculate figure of merit. We assume a simulation has already
            # been run and the fields are contained in the sim object which is
            # of type FDFD
            ...
            return fom

        def calc_dFdx(self, sim, params):
            # calculate derivative of F with respect to fields which is used as
            # source in adjoint simulation
            ...
            return dFdx

        def calc_grad_p(self, sim, params):
            # calculte the gradient of F with respect to the design variables,
            # holding the fields constant
            ...
            return grad_y

Here :meth:`.AdjointMethod.update_system` updates the the system based on the
current set of design parameters, :meth:`.AdjointMethod.calc_fom` calculates
the value of F(\\mathbf{E}, \\mathbf{H}, y_1, y_2, \cdots, y_N) for the specified
set of design parameters in :samp:`params`, :meth:`.AdjointMethod.calc_dFdx`
calculates the derivative of :math:`F` with respect to the relevant field
components, and :meth:`.AdjointMethod.calc_grad_y` calculates the gradient of F
with respect to the non-field-dependent quantities in F.

In order to verify that the :meth:`.AdjointMethod.calc_fom`,
:meth:`.AdjointMethod.calc_dFdx`, and :meth:`.AdjointMethod.calc_grad_y`
functions are consistent, the gradient accuracty should always be verified. The
AdjointMethod base class defines a function to do just this.  For example, using
the :samp:`MyAdjointMethod` that we have just defined, we might do:

.. doctest::

    # set up an FDFD simulation object called 'sim'
    ...

    # create the adjoint method object
    am = MyAdjointMethod(sim, myparam)

    # check the gradient
    init_params = ...
    am.check_gradient(init_params, indices=np.arange(0,10))

In this example, we check the accuracy of the gradients computed for a given
initial set of design parameters called :samp:`init_params`.  We restrict the
check to the first ten components of the gradient in order to speed things up.

In addition to the adjoint method base class, there are a number of
application-specific implementations which you may find useful. In particular,
the :class:`.AdjointMethodFM2D` class provides a simplified interface for
computing the gradient of a function that depends not only on the fields but
also the permittivity and permeability.  In addition to the functions specified
above, the user must implement an additional function
:meth:`.AdjointMethodFM.calc_dFdm` which must compute the derivative of the figure
of merit :math:`F` with respect to the permittivity and permeability,
:math:`\epsilon` and :math:`\mu`.  An example of such a function would be, for
example, the total absorption of electromagnetic energy in a domain.

Furthermore, in electromagnetics, efficiencies make common figures of merit.
In many cases, this efficiency is defined in terms of the ratio of a calculated
power to the total source power of the system.  Because differentiating these
power-normalized quantities (which depend on the fields and the
permittivity/permeability) is rather laborious, this functionality is
implemented in the :class:`.AdjointMethodPNF2D` and
:class:`.AdjointMethodPNF3D` classes for convenience.

See Also
--------
emopt.fdfd.FDFD : Base class for simulators supported by :class:`.AdjointMethod`.

emopt.optimizer.Optimizer : Primary application of :class:`.AdjointMethod` to optimization.

References
----------
.. [1] A. Michaels and E. Yablonovitch, "Leveraging continuous material averaging for inverse electromagnetic design," Opt. Express 26, 31717-31737 (2018)
"""
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
from builtins import object
from . import fdfd
from . import fdtd
from .misc import info_message, warning_message, error_message, RANK, \
NOT_PARALLEL, run_on_master, N_PROC, COMM, DomainCoordinates
from . import fomutils
from .modes import Kahan_dot

import numpy as np
from math import pi
from abc import ABCMeta, abstractmethod
from petsc4py import PETSc
from mpi4py import MPI
from future.utils import with_metaclass
from timeit import default_timer as timer
from datetime import datetime
import gc

import ctypes
import ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library('c'))

__author__ = "Andrew Michaels"
__license__ = "BSD-3"
__version__ = "2023.1.16"
__maintainer__ = "Peng Sun"
__status__ = "development"

class AdjointMethodEigen(with_metaclass(ABCMeta, object)):
    """Adjoint Method Class

    Defines the core functionality needed to compute the gradient of a function
    of the form

    .. math:
        F \\rightarrow F(\\mathbf{E}, \\mathbf{H}, \\vec{p})

    with respect to an arbitrary set of design variables :math:`\\vec{p}`.
    In general, the gradient is given by

    .. math::
        \\nabla F = \\nabla_\mathrm{AM} F + \\frac{\partial F}{\partial \\vec{p}}

    where :math:`\\nabla_\\mathrm{AM} F` is the gradient of :math:`F` computed
    using the adjoint method, and the remaining gradient term corresponds to
    any explicit dependence of the figure of merit on the design parameters.
    The derivatives of these quantities are assumed to be known and should be
    computed using :meth:`.AdjointMethod.calc_grad_p` function.

    In order to use the AdjointMethod class, it should extended and the
    abstract methods :meth:`.AdjointMethod.update_system`,
    :meth:`.AdjointMethod.calc_fom`, :meth:`.AdjointMethod.calc_dFdx`, and
    :meth:`.AdjointMethod.calc_grad_p` should be implemented for the desired
    application.

    Notes
    -----
    Currently source derivatives are not supported.  If needed, this should not
    be too difficult to achieve by extending :class:`.AdjointMethod`

    Parameters
    ----------
    sim : emopt.simulation.MaxwellSolver
        Simulation object
    step : float
        Step sized used in the calculation of :math:`\partial A / \partial p_i`

    Attributes
    ----------
    sim : emopt.simulation.MaxwellSolver
        Simulation object
    step : float
        Step sized used in the calculation of :math:`\partial A / \partial p_i`

    Methods
    -------
    update_system(params)
        **(ABSTRACT METHOD)** Update the geometry of the system.
    calc_fom(sim, params)
        **(ABSTRACT METHOD)** Calculate the figure of merit.
    calc_dFdx(sim, params)
        **(ABSTRACT METHOD)** Calculate the derivative of the figure of merit
        with respect to the electric and magnetic field vectors.
    get_update_boxes(sim, params)
        Define update boxes which specify which portion of the underlying
        spatial grid is modified by each design variable.
    fom(params)
        Get the figure of merit.
    calc_gradient(sim, params)
        Calculate the figure of merit in a general way.
    gradient(params)
        Get the gradient for at the current set of design parameter values.
    """

    def __init__(self, modes, step=1e-3):
        self.modes = modes
        self.prev_params = []
        self._step = step
        self._UseAutoDiff = False

    @property
    def step(self):
        """
        Step size used for numerical differentiation of :math:`A`

        :getter: Returns the step size.
        :setter: Sets the step size
        :type: float
        """
        return self._step

    @step.setter
    def step(self, val):
        if(np.abs(val) > self.modes.dx/1e3):
            if(NOT_PARALLEL):
                warning_message('Step size used for adjoint method may be too '
                                'large.  Consider reducing it to ~1e-3*dx')
        self._step = val

    @abstractmethod
    def update_system(self, params):
        """Update the geometry/material distributions of the system.

        In order to calculate the gradient of a figure of merit, we need to
        define a mapping between the abstract design parameters of the system
        (which are contained in the vector :samp:`params`) and the underlying
        geometry or material distribution which makes up the physical system.
        We define this mapping here.

        Notes
        -----
        In general, calculation of the gradient involves calling this function
        once per design variable.  In other words, if :samp:`len(params)` is
        equal to N, then this method is called at least N times in order to
        calculate the gradient.  For cases where N is large, it is recommended
        an effort be made to avoid performing very costly operations in this
        method.

        Parameters
        ----------
        params : numpy.ndarray
            1D array containing design parameter values (one value per design
            parameter)
        """
        pass

    @abstractmethod
    def calc_fom(self, modes, params):
        """Calculate the figure of merit.

        Notes
        -----
        This function is called by the :func:`.AdjointMethod.fom` function. In
        this case, update_system(params) and sim.solve_forward() are guaranteed
        to be called before this function is executed.

        If this function is called outside of the :func:`.AdjointMethod.fom`
        function (which is not advised), it is up to the caller to ensure that
        the :func:`.emopt.FDFD.solve_forward` has been run previously.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            Simulation object
        params : numpy.ndarray
            1D vector containing design parameter values.
        """
        pass

    @abstractmethod
    def calc_dFdx(self, modes, params):
        """Calculate the derivative of the figure of merit with respect to the
        vector containing the electric and magnetic fields.

        In order to calculate the gradient of the figure of merit, an adjoint
        simulation must first be run.  The sources in the adjoint simulation are
        given by :math:`\partial F / \partial x` where :math:`F` is the figure of
        merit and :math:`x` is a vector containing the electric and magnetic fields
        contained on a discreter grid.  Because we are differentiating with respect
        to a vector, the resulting derivative will also be a vector.

        This function must be overriden and implemented to calculate the derivative
        of the figure of merit defined in :func:`calc_fom`.

        The exact format of :math:`x` depends on the exact type of
        :class:`emopt.fdfd.FDFD` object which generated it.  Consult
        :mod:`emopt.fdfd` for details.

        See Also
        --------
        emopt.fdfd.FDFD : Base class for simulators which generate :math:`x`
        """
        pass

    @abstractmethod
    def calc_dFdn(self, modes, params):
        """Calculate the derivative of the figure of merit with respect to the
        vector containing the electric and magnetic fields.

        In order to calculate the gradient of the figure of merit, an adjoint
        simulation must first be run.  The sources in the adjoint simulation are
        given by :math:`\partial F / \partial x` where :math:`F` is the figure of
        merit and :math:`x` is a vector containing the electric and magnetic fields
        contained on a discreter grid.  Because we are differentiating with respect
        to a vector, the resulting derivative will also be a vector.

        This function must be overriden and implemented to calculate the derivative
        of the figure of merit defined in :func:`calc_fom`.

        The exact format of :math:`x` depends on the exact type of
        :class:`emopt.fdfd.FDFD` object which generated it.  Consult
        :mod:`emopt.fdfd` for details.

        See Also
        --------
        emopt.fdfd.FDFD : Base class for simulators which generate :math:`x`
        """
        pass

    @abstractmethod
    def calc_grad_p(self, modes, params):
        """Compute the gradient of of the figure of merit with respect to the
        design variables :math:`\\vec{p}`, **holding the fields
        constant**.

        This function should calculate the list of partial derivatives
        of the figure of merit with respect to each design variable

        .. math:
            \\frac{\\partial F}{\\partial \\vec{p}} =
            \\left[\\frac{\\partial F}{\\partial p_1}, \\frac{\\partial
            F}{\\partial p_2}, \\cdots, \\frac{\\partial F}{\\partial p_N}\\right]

        This allows us to include an explicit dependence on the design
        variables in our figure of merit. This is useful for imposing
        constraints in an optimization.

        Notes
        -----
        This function is executed in parallel on all nodes.  If execution on
        master node is desired, you can either apply the @run_on_master
        decorator or use if(NOT_PARALLEL).

        Parameters
        ----------
        sim : emoptg.fdfd.FDFD
            The FDFD object
        params : numpy.ndarray
            The array containing the current set of design parameters

        Returns
        -------
        numpy.ndarray
            The partial derivatives with respect to the design variables.
        """
        pass

    # @abstractmethod
    def calc_gradient_manual(self, modes, params):
        pass

    def get_update_boxes(self, modes, params):
        """Get update boxes used to speed up the updating of A.

        In order to compute the gradient, we need to calculate how A changes
        with respect to modification of the design variables.  This generally
        requires updating the material values in A.  We can speed this process
        up by only updating the part of the system which is affect by the
        modification of each design variable.

        By default, the update boxes cover the whole simulation area.  This
        method can be overriden in order to modify this behavior.

        Parameters
        ----------
        sim : FDFD
            Simulation object.  sim = self.sim
        params : numpy.array or list of floats
            List of design parameters.

        Returns
        -------
            Either a list of tuples or a list of lists of tuples containing
            (xmin, xmax, ymin, ymax) in 2D and (xmin, xmax, ymin, ymax, zmin,
            zmax) in 3D which describes which portion of the system should be
            update during gradient calculations.
        """
        X = modes._M * modes.dx
        Y = modes._N * modes.dy
        lenp = len(params)
        return [(0, X, 0, Y) for i in range(lenp)]

    def fom(self, params):
        """Run a forward simulation and calculate the figure of merit.

        Notes
        -----
        The simualtion is performed in parallel with the help of all of the
        MPI node, however the calculation of the figure of merit itself is
        currently only performed on the master node (RANK == 0)

        Parameters
        ----------
        params : numpy.ndarray
            List of design parameters of the system

        Returns
        -------
        float
            **(Master node only)** The figure of merit :math:`F(\mathbf{E},\mathbf{H} ; \mathbf{p})`
        """
        # update the system using the design parameters
        self.update_system(params)
        self.prev_params = params

        self.modes.build()
        self.modes.solve()
        return self.calc_fom(self.modes, params)

    def solve_adjoint(self, params):
        ### Solve the adjoint eigenproblem of (A-nB)*lam=(I-xy^H*B)*dFdx^H
        dFdx = self.calc_dFdx(params)
        dFdx = np.conj(dFdx)
        if np.linalg.norm(dFdx) < 1E-12:
            ### g=0 -> lam=0
            lam = PETSc.Vec().createWithArray(np.zeros_like(dFdx))
            self.lam0 = lam
        else:
            g = self.modes._B.createVecRight()
            g.setArray(dFdx.copy())
            Bg = self.modes._B.createVecRight()
            self.modes._B.mult(g, Bg)
            alpha = self.modes._y[self.modeidx].dot(Bg)
            M = self.modes._A.duplicate(copy=True)
            M.axpy(-self.modes.neff[self.modeidx], self.modes._B, 
                   structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)   ### M = A-nB

            xalpha = self.modes._x[self.modeidx].duplicate()
            self.modes._x[self.modeidx].copy(xalpha)
            xalpha.scale(alpha)
            Pg = g.duplicate()
            Pg.waxpy(-1.0, xalpha, g)   ### g - x*alpha
            denom = self.modes._x[self.modeidx].dot(self.modes._x[self.modeidx])
            beta = self.modes._x[self.modeidx].dot(Pg) / denom
            Pg.axpy(-beta, self.modes._x[self.modeidx])

            nullspace = PETSc.NullSpace().create(vectors=[self.modes._x[self.modeidx]])
            try:
                M.setNullSpace(nullspace)
            except Exception:
                pass

            ksp = PETSc.KSP().create(self.modes._A.getComm())
            ksp.setOperators(M)
            ksp.setType('preonly')
            ksp.getPC().setType('lu')
            ksp.getPC().setFactorSolverType('mumps')
            ksp.setFromOptions()

            lam = g.duplicate()
            ksp.solve(Pg, lam)
            its = ksp.getIterationNumber()
            reason = ksp.getConvergedReason()
            PETSc.Sys.Print(f"KSP converged in {its} iterations with reason {reason}")

            yH_lam = self.modes._y[self.modeidx].dot(lam)
            yH_y = self.modes._y[self.modeidx].dot(self.modes._y[self.modeidx])
            lam0 = lam.duplicate()
            lam.copy(lam0)
            lam0.axpy(-yH_lam/yH_y, self.modes._y[self.modeidx])
            self.lam0 = lam0

    def calc_gradient(self, params):
        """ Calculate dF/dx*dx/dp + dF/dn*dn/dp in the same method
        
        dn/dp is calculated via Hellman-Feynman theorem: y^H*dA/dp*x
        dF/dx*dx/dp is calculated via the adjoint problem of (A-nB)*lam=(I-xy^H)*dF/dx^H

        dn/dp is computed with Hellman-Feynman theorem: y^H * dA/dp * x:
        math:`A` with respect to the design parameters of
        the system, i.e. :math:`\partial A / \partial p_i`. In the most general
        case, we can compute this derivative using finite differences. This
        involves perturbing each design variable of the system by a small
        amount one at a time and updating :math:`A`.  Doing so allows us to
        approximate the derivative as

        .. math::
            \\frac{\partial A}{\partial p_i} \\approx \\frac{A(p_i + \Delta p) - A(p_i)}{\Delta p}

        So long as :math:`\Delta p` is small enough, this approximation is
        quite accurate.

        This function handles this process.

        Notes
        -----
        1. This function is called after the forward and adjoint simulations have
        been executed.

        2. Technically, a centered difference would be more accurate, however
        this whole implementation relies on mesh smoothing which allows us to
        make very small steps :math:`\Delta p` and thus in reality, the benefit
        is negligable.

        Parameters
        ----------
        sim : FDFD
            Simulation object.  sim = self.sim
        params : numpy.array or list of floats
            List of design parameters.

        Returns
        -------
        numpy.array
            **(Master node only)** Gradient of figure of merit, i.e. list of
            derivatives of fom with respect to each design variable
        """
        # get the current diagonal elements of A.
        # only these elements change when the design variables change.

        step = self._step
        lenp = len(params)
        self.params = params

        grad_full = None
        grad_parts = []
        if(RANK == 0):
            grad_full = np.zeros(N_PROC, dtype=np.double)

        A0 = self.modes._A.copy()
        dFdn = self.calc_dFdn(params)
        gradient = np.zeros(lenp)
        for ii in range(lenp):
            print("param",ii+1,"of",lenp, datetime.now().isoformat()+'\033[1A\r')
            p0 = params[ii]

            # perturb the system
            params[ii] += step
            self.update_system(params)
            self.modes.build()

            #### dA/dp = (A-A0)/step
            dAdp = self.modes._A.copy()
            dAdp.axpy(-1.0, A0, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
            dAdp.scale(1.0/step)

            ### Hellman-Feynman theorem: dn/dp = y^H dA x
            tmpvec = dAdp.createVecRight()
            dAdp.mult(self.modes._x[self.modeidx], tmpvec)
            dndp = np.real(Kahan_dot(self.modes._y[self.modeidx], tmpvec))

            ### dFdx * dxdp
            dFdx_dxdp = np.real(Kahan_dot(self.lam0, tmpvec))
            ### dFdx * dxdp + dFdn * dndp
            grad_parts.append(dFdx_dxdp + dFdn * dndp)

            # # revert the parameters
            params[ii] = p0

        ### revert the system to original
        self.update_system(params)
        self.modes.build()

        COMM.Barrier()
        for ii in range(lenp):
            grad_full = COMM.gather(grad_parts[ii], root=0)
            if(NOT_PARALLEL):
                gradient[ii] = np.sum(grad_full)

        if(NOT_PARALLEL):
            return gradient

    def gradient(self, params):
        """Manage the calculation of the gradient figure of merit.

        To calculate the gradient, we update the system, run a forward and
        adjoint simulation, and then calculate the gradient using
        :func:`calc_gradient`.  Most of these operations are done in parallel
        using MPI.

        Parameters
        ----------
        params : numpy.ndarray
            List of design parameters of the system

        Returns
        -------
        numpy.ndarray
            (Master node only) The gradient of the figure of merit computed  with 
            respect to the design variables
        """
        # update system
        print('Update system:'+datetime.now().isoformat())
        self.update_system(params)
        self.modes.build()

        # Solve the forward problem
        if(not np.array_equal(self.prev_params, params)):
            self.modes.solve()

        # Solve the adjoint problem
        print('Solve adjoint problem:'+datetime.now().isoformat())
        self.solve_adjoint(params)

        if(NOT_PARALLEL):
            info_message('Calculating gradient...')

        grad_f = self.calc_gradient(params)
        grad_p = self.calc_grad_p(params)

        if(NOT_PARALLEL):
            self.current_gradient = grad_f
            return grad_f + grad_p
        else:
            return None

    def check_gradient(self, params, indices=[], plot=True, verbose=True,
                       return_gradients=False):
        """Verify that the gradient is accurate.

        It is highly recommended that the accuracy of the gradients be checked
        prior to being used. If the accuracy is above ~1%, it is likely that
        there is an inconsistency between how the figure of merit and adjoint
        sources (dFdx) are being computed.

        The adjoint method gradient error is evaluated by comparing the
        gradient computed using the adjoint method to a gradient computed by
        direct finite differences.  In other words, the "correct" derivatives
        to which the adjoint method gradient is compared are given by

        .. math::
            \\frac{\partial F}{\partial p_i} \\approx \\frac{F(p_i + \Delta p) - F(p_i)}{\Delta p}

        Note that this method for calculating the gradient is not used in a
        practical setting because it requires performing N+1 simulations in
        order to compute the gradient with respect to N design variables
        (compared to only 2 simulations in the case of the adjoint method).

        Parameters
        ----------
        params : numpy.ndarray
            design parameters
        indices : list or numpy.ndarray
            list of gradient indices to check. An empty list indicates that the
            whole gradient should be verified. A subset of indices may be
            desirable for large problems.  (default = [])
        plot : bool (optional)
            Plot the gradients and errors (default = True)
        verbose : bool (optional)
            If True, print progress (default = True)
        return_gradients : bool
            If True, return the gradient arrays (default = False)

        Returns
        -------
        float
            Relative error in gradient.
        """

        if(indices == []):
            indices = np.arange(0, len(params),1)

        # make sure everything is up to date
        self.update_system(params)
        self.modes.build()

        grad_am = self.gradient(params)
        grad_fd = np.zeros(len(indices))

        fom0 = self.fom(params)
        fd_step = self.step
        # calculate the "true" derivatives using finite differences
        if(NOT_PARALLEL and verbose):
            info_message('Checking gradient...')

        for i in range(len(indices)):
            if(NOT_PARALLEL and verbose):
                print('\tDerivative %d of %d' % (i+1, len(indices)))

            j = indices[i]
            p0 = params[j]
            params[j] += fd_step
            fom1 = self.fom(params)
            if(NOT_PARALLEL):
                grad_fd[i] = (fom1-fom0)/fd_step
            params[j] = p0

        ### revert system to original states
        self.update_system(params)
        self.modes.build()
        self.modes.solve()

        if(NOT_PARALLEL):
            errors = np.abs(grad_fd - grad_am[indices]) / np.abs(grad_fd)
            error_tot = np.linalg.norm(grad_fd - grad_am[indices]) / np.linalg.norm(grad_fd)

            if(error_tot < 0.01 and verbose):
                info_message('The total error in the gradient is %0.4E' % \
                             (error_tot))
            else:
                warning_message('The total error in the gradient is %0.4E '
                                'which is over 1%%' % (error_tot), \
                                'emopt.adjoint_method')

            if(plot):
                import matplotlib.pyplot as plt
                f = plt.figure()
                ax1 = f.add_subplot(311)
                ax2 = f.add_subplot(312)
                ax3 = f.add_subplot(313)

                xs = np.arange(len(indices))
                ax1.bar(xs, grad_fd)
                ax1.set_title('Finite Differences')
                ax2.bar(xs, grad_am[indices])
                ax2.set_title('Adjoint Method')
                ax3.bar(xs, errors)
                ax3.set_title('Error in Adjoint Method')

                for ax in [ax1, ax2, ax3]:
                    ax.set_xticklabels(['%d' % i for i in indices])

                ax3.set_yscale('log', nonposy='clip')

                plt.savefig('Check_gradient.png', dpi=600)

            if(return_gradients):
                return error_tot, grad_fd, grad_am
            else:
                return error_tot
        else:
            if(return_gradients):
                return None, None, None
            return None

class AdjointMethodEigenMO(with_metaclass(ABCMeta, AdjointMethodEigen)):
    """An AdjointMethod object for an ensemble of different figures of merit
    (Multi-objective adjoint method).

    In many situations, it is desirable to calculate the sensitivities of a
    structure corresponding to multiple objective functions.  A simple common
    exmaple of this a broadband figure of merit which considers the performance
    of structure at a range of different excitation frequencies/wavelengths.
    In other cases, it may be desirable to calculate a total sensitivity which
    is made up of two different figures of merit which are calculated for the
    same excitation.

    In either case, we need a way to easily handle these more complicated
    figures of merits and their gradients (i.e. the sensitivities). This class
    provides a simple interface to do just that.  By overriding calc_total_fom
    and calc_total_gradient, you can build up more complicated figures of
    merit.

    Parameters
    ----------
    ams : list of :class:`.AdjointMethod`
        A list containing *extended* AdjointMethod objects

    Attributes
    ----------
    adjoint_methods : list of :class:`.AdjointMethod`
        A list containing extended AdjointMethod objects
    """

    def __init__(self, ams, step=1e-6):
        self._ams = ams
        self._foms_current = np.zeros(len(ams))
        self._step = step

    @property
    def adjoint_methods(self):
        return self._ams

    @adjoint_methods.setter
    def adjoint_methods(self, new_ams):
        self._ams = new_ams

    def update_system(self, params):
        """Update all of the individual AdjointMethods."""
        for am in self._ams:
            am.update_system(params)

    def calc_fom(self, modes, params):
        """Calculate the figure of merit.
        """
        # this just redirects to calc_total_foms
        return self.calc_total_fom(self._foms_current)

    def calc_dFdx(self, modes, params):
        pass

    def calc_dFdn(self, modes, params):
        pass

    def calc_grad_p(self, modes, params):
        pass

    @abstractmethod
    def calc_total_fom(self, foms):
        """Calculate the 'total' figure of merit based on a list of evaluated
        objective functions.

        The user should override this function in order to define how all of the
        individual figures of merit are combined to form a single 'total'
        figure of merit. This may be a sum of the input FOMs, a minimax of the
        FOMs, etc.  A common example is to combine figures of merit calculated
        for different wavelengths of operation.

        See Also
        --------
        :ref:`emopt.fomutils` : functions which may be useful for combining figures of merit

        Parameters
        ----------
        foms : list of float
            List containing individual FOMs which are used to compute the total
            figure of merit.

        Returns
        -------
        float
            The total figure of merit
        """
        pass

    @abstractmethod
    def calc_total_gradient(self, foms, grads):
        """Calculate the 'total' gradient of a figure of merit based on a list
        of evaluated objective functions.

        The user should override this function in order to define the gradient
        of the total figure of merit.

        See Also
        --------
        :ref:`emopt.fomutils` : functions which may be useful for combining figures of merit and their gradients.

        Parameters
        ----------
        foms : list
            List of individual foms
        grads : list
            List of individual grads

        Returns
        -------
        numpy.ndarray
            1D numpy array containing total gradient. note: the output vector
            should have the same shape as the input vectors contained in grads
        """
        pass

    def fom(self, params):
        """Calculate the total figure of merit.

        Notes
        -----
        Overrides :class:`.AdjointMethod`.fom(...)

        Parameters
        ----------
        params : numpy.ndarray
            Design parameters

        Returns
        -------
        float
            (Master node only) The total figure of merit
        """
        foms = []

        for am in self._ams:
            foms.append(am.fom(params))

        self._foms_current = foms
        if(NOT_PARALLEL):
            fom_total = self.calc_total_fom(foms)
            return fom_total
        else:
            return None

    def gradient(self, params):
        """Calculate the total gradient.

        Notes
        -----
        Overrides :class:`.AdjointMethod`.gradient(...)

        Parameters
        ----------
        params : numpy.ndarray
            Design parameters with respect to which gradient is evaluated

        Returns
        -------
        numpy.ndarray
            (Master node only) The gradient of total figure of merit.
        """
        foms = []
        grads = []

        for am in self._ams:
            grads.append( am.gradient(params) )
            foms.append( am.calc_fom(am.modes, params) )

        if(NOT_PARALLEL):
            grad_total = self.calc_total_gradient(foms, grads)
            return grad_total
        else:
            return None

class AdjointMethod(with_metaclass(ABCMeta, object)):
    """Adjoint Method Class

    Defines the core functionality needed to compute the gradient of a function
    of the form

    .. math:
        F \\rightarrow F(\\mathbf{E}, \\mathbf{H}, \\vec{p})

    with respect to an arbitrary set of design variables :math:`\\vec{p}`.
    In general, the gradient is given by

    .. math::
        \\nabla F = \\nabla_\mathrm{AM} F + \\frac{\partial F}{\partial \\vec{p}}

    where :math:`\\nabla_\\mathrm{AM} F` is the gradient of :math:`F` computed
    using the adjoint method, and the remaining gradient term corresponds to
    any explicit dependence of the figure of merit on the design parameters.
    The derivatives of these quantities are assumed to be known and should be
    computed using :meth:`.AdjointMethod.calc_grad_p` function.

    In order to use the AdjointMethod class, it should extended and the
    abstract methods :meth:`.AdjointMethod.update_system`,
    :meth:`.AdjointMethod.calc_fom`, :meth:`.AdjointMethod.calc_dFdx`, and
    :meth:`.AdjointMethod.calc_grad_p` should be implemented for the desired
    application.

    Notes
    -----
    Currently source derivatives are not supported.  If needed, this should not
    be too difficult to achieve by extending :class:`.AdjointMethod`

    Parameters
    ----------
    sim : emopt.simulation.MaxwellSolver
        Simulation object
    step : float
        Step sized used in the calculation of :math:`\partial A / \partial p_i`

    Attributes
    ----------
    sim : emopt.simulation.MaxwellSolver
        Simulation object
    step : float
        Step sized used in the calculation of :math:`\partial A / \partial p_i`

    Methods
    -------
    update_system(params)
        **(ABSTRACT METHOD)** Update the geometry of the system.
    calc_fom(sim, params)
        **(ABSTRACT METHOD)** Calculate the figure of merit.
    calc_dFdx(sim, params)
        **(ABSTRACT METHOD)** Calculate the derivative of the figure of merit
        with respect to the electric and magnetic field vectors.
    get_update_boxes(sim, params)
        Define update boxes which specify which portion of the underlying
        spatial grid is modified by each design variable.
    fom(params)
        Get the figure of merit.
    calc_gradient(sim, params)
        Calculate the figure of merit in a general way.
    gradient(params)
        Get the gradient for at the current set of design parameter values.
    """

    def __init__(self, sim, step=1e-8):
        self.sim = sim
        self.prev_params = []
        self._step = step
        self._UseAutoDiff = False

    @property
    def step(self):
        """
        Step size used for numerical differentiation of :math:`A`

        :getter: Returns the step size.
        :setter: Sets the step size
        :type: float
        """
        return self._step

    @step.setter
    def step(self, val):
        if(np.abs(val) > self.sim.dx/1e3):
            if(NOT_PARALLEL):
                warning_message('Step size used for adjoint method may be too '
                                'large.  Consider reducing it to ~1e-3*dx')
        self._step = val

    @abstractmethod
    def update_system(self, params):
        """Update the geometry/material distributions of the system.

        In order to calculate the gradient of a figure of merit, we need to
        define a mapping between the abstract design parameters of the system
        (which are contained in the vector :samp:`params`) and the underlying
        geometry or material distribution which makes up the physical system.
        We define this mapping here.

        Notes
        -----
        In general, calculation of the gradient involves calling this function
        once per design variable.  In other words, if :samp:`len(params)` is
        equal to N, then this method is called at least N times in order to
        calculate the gradient.  For cases where N is large, it is recommended
        an effort be made to avoid performing very costly operations in this
        method.

        Parameters
        ----------
        params : numpy.ndarray
            1D array containing design parameter values (one value per design
            parameter)
        """
        pass

    @abstractmethod
    def calc_fom(self, sim, params):
        """Calculate the figure of merit.

        Notes
        -----
        This function is called by the :func:`.AdjointMethod.fom` function. In
        this case, update_system(params) and sim.solve_forward() are guaranteed
        to be called before this function is executed.

        If this function is called outside of the :func:`.AdjointMethod.fom`
        function (which is not advised), it is up to the caller to ensure that
        the :func:`.emopt.FDFD.solve_forward` has been run previously.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            Simulation object
        params : numpy.ndarray
            1D vector containing design parameter values.
        """
        pass

    @abstractmethod
    def calc_dFdx(self, sim, params):
        """Calculate the derivative of the figure of merit with respect to the
        vector containing the electric and magnetic fields.

        In order to calculate the gradient of the figure of merit, an adjoint
        simulation must first be run.  The sources in the adjoint simulation are
        given by :math:`\partial F / \partial x` where :math:`F` is the figure of
        merit and :math:`x` is a vector containing the electric and magnetic fields
        contained on a discreter grid.  Because we are differentiating with respect
        to a vector, the resulting derivative will also be a vector.

        This function must be overriden and implemented to calculate the derivative
        of the figure of merit defined in :func:`calc_fom`.

        The exact format of :math:`x` depends on the exact type of
        :class:`emopt.fdfd.FDFD` object which generated it.  Consult
        :mod:`emopt.fdfd` for details.

        See Also
        --------
        emopt.fdfd.FDFD : Base class for simulators which generate :math:`x`
        """
        pass

    @abstractmethod
    def calc_grad_p(self, sim, params):
        """Compute the gradient of of the figure of merit with respect to the
        design variables :math:`\\vec{p}`, **holding the fields
        constant**.

        This function should calculate the list of partial derivatives
        of the figure of merit with respect to each design variable

        .. math:
            \\frac{\\partial F}{\\partial \\vec{p}} =
            \\left[\\frac{\\partial F}{\\partial p_1}, \\frac{\\partial
            F}{\\partial p_2}, \\cdots, \\frac{\\partial F}{\\partial p_N}\\right]

        This allows us to include an explicit dependence on the design
        variables in our figure of merit. This is useful for imposing
        constraints in an optimization.

        Notes
        -----
        This function is executed in parallel on all nodes.  If execution on
        master node is desired, you can either apply the @run_on_master
        decorator or use if(NOT_PARALLEL).

        Parameters
        ----------
        sim : emoptg.fdfd.FDFD
            The FDFD object
        params : numpy.ndarray
            The array containing the current set of design parameters

        Returns
        -------
        numpy.ndarray
            The partial derivatives with respect to the design variables.
        """
        pass

    # @abstractmethod
    def calc_gradient_manual(self, sim, params):
        pass

    def get_update_boxes(self, sim, params):
        """Get update boxes used to speed up the updating of A.

        In order to compute the gradient, we need to calculate how A changes
        with respect to modification of the design variables.  This generally
        requires updating the material values in A.  We can speed this process
        up by only updating the part of the system which is affect by the
        modification of each design variable.

        By default, the update boxes cover the whole simulation area.  This
        method can be overriden in order to modify this behavior.

        Parameters
        ----------
        sim : FDFD
            Simulation object.  sim = self.sim
        params : numpy.array or list of floats
            List of design parameters.

        Returns
        -------
            Either a list of tuples or a list of lists of tuples containing
            (xmin, xmax, ymin, ymax) in 2D and (xmin, xmax, ymin, ymax, zmin,
            zmax) in 3D which describes which portion of the system should be
            update during gradient calculations.
        """
        if(sim.ndims == 2):
            X = sim.X
            Y = sim.Y
            lenp = len(params)
            return [(0, X, 0, Y) for i in range(lenp)]
        elif(sim.ndims == 3):
            X = sim.X
            Y = sim.Y
            Z = sim.Z
            lenp = len(params)
            return [(0,X,0,Y,0,Z) for i in range(lenp)]

    def fom(self, params):
        """Run a forward simulation and calculate the figure of merit.

        Notes
        -----
        The simualtion is performed in parallel with the help of all of the
        MPI node, however the calculation of the figure of merit itself is
        currently only performed on the master node (RANK == 0)

        Parameters
        ----------
        params : numpy.ndarray
            List of design parameters of the system

        Returns
        -------
        float
            **(Master node only)** The figure of merit :math:`F(\mathbf{E},\mathbf{H} ; \mathbf{p})`
        """
        # update the system using the design parameters
        self.update_system(params)
        self.prev_params = params
        self.sim.update()

        #run the forward sim
        self.sim.solve_forward()

        # calculate the figure of merit
        return self.calc_fom(self.sim, params)

    def calc_gradient(self, sim, params):
        """Calculate the gradient of the figure of merit.

        The gradient of the figure of merit is computed by running a forward
        simulation, adjoint simulation, and then computing the derivatives of
        the system matrix :math:`A` with respect to the design parameters of
        the system, i.e. :math:`\partial A / \partial p_i`. In the most general
        case, we can compute this derivative using finite differences. This
        involves perturbing each design variable of the system by a small
        amount one at a time and updating :math:`A`.  Doing so allows us to
        approximate the derivative as

        .. math::
            \\frac{\partial A}{\partial p_i} \\approx \\frac{A(p_i + \Delta p) - A(p_i)}{\Delta p}

        So long as :math:`\Delta p` is small enough, this approximation is
        quite accurate.

        This function handles this process.

        Notes
        -----
        1. This function is called after the forward and adjoint simulations have
        been executed.

        2. Technically, a centered difference would be more accurate, however
        this whole implementation relies on mesh smoothing which allows us to
        make very small steps :math:`\Delta p` and thus in reality, the benefit
        is negligable.

        Parameters
        ----------
        sim : FDFD
            Simulation object.  sim = self.sim
        params : numpy.array or list of floats
            List of design parameters.

        Returns
        -------
        numpy.array
            **(Master node only)** Gradient of figure of merit, i.e. list of
            derivatives of fom with respect to each design variable
        """
        # get the current diagonal elements of A.
        # only these elements change when the design variables change.

        # del self.sim._Ex; del self.sim._Ey; del self.sim._Ez
        # del self.sim._Hx; del self.sim._Hy; del self.sim._Hz
        gc.collect()

        if self._UseAutoDiff is True:
            import torch
            torch.set_default_dtype(torch.double)
            torch.set_default_tensor_type(torch.DoubleTensor)
            
            # ksigz = self.ksigz
            paramdiff = torch.zeros(len(params))
            for ii in range(len(params)):
                paramdiff[ii] = params[ii]

            update_boxes = self.get_update_boxes(sim, params)
            ub = update_boxes[0]
            bbox = DomainCoordinates(ub[0], ub[1], ub[2], ub[3], ub[4], ub[5], self.sim._dx, self.sim._dy, self.sim._dz)
            g_inds, l_inds, d_inds, sizes = self.sim._FDTD__get_local_domain_overlap(bbox)
            # lower/upper surfaces of the etching
            zmin, zmax = self.sim.perturb_zmin, self.sim.perturb_zmax
            gradient = np.zeros(len(params))

            # Ex
            print("Calc Ex gradient:"+datetime.now().isoformat())
            sx, sy, sz = 0.5*self.sim._dx, 0.0*self.sim._dy, -0.5*self.sim._dz
            z = torch.linspace(sz+l_inds[0]*self.sim.dz, sz+(l_inds[0]+sizes[0])*self.sim.dz, sizes[0])
            shape = self.update_system_diffgeo(paramdiff, sx=sx, sy=sy)
            Z = shape.unsqueeze(-1).expand(-1,-1,z.shape[0])
            # Z = Z * torch.sigmoid(ksigz*(z-zmin))*torch.sigmoid(-ksigz*(z-zmax)).unsqueeze(0).unsqueeze(0).expand(shape.shape[0], shape.shape[1], -1)
            # Z += torch.sigmoid(-ksigz*(z-zmin)).unsqueeze(0).unsqueeze(0).expand(shape.shape[0], shape.shape[1], -1)
            Z = Z * torch.clamp((z-zmin)/sim._dz+0.5, min=0.0, max=1.0)*torch.clamp(-(z-zmax)/sim._dz+0.5, min=0.0, max=1.0).unsqueeze(0).unsqueeze(0).expand(shape.shape[0], shape.shape[1], -1)
            Z += torch.clamp(-(z-zmin)/sim._dz+0.5, min=0.0, max=1.0).unsqueeze(0).unsqueeze(0).expand(shape.shape[0], shape.shape[1], -1)

            Z = self.sim.perturb_epsclad + (self.sim.perturb_epscore-self.sim.perturb_epsclad) * Z
            Zcomp = torch.complex(Z, torch.zeros_like(Z)).permute(2,1,0)
            fields_tensor = torch.tensor(1j*self.sim._Ex_fwd_t0_pbox[...]*self.sim._Ex_adj_t0_pbox[...])
            loss_grad = torch.dot(torch.flatten(fields_tensor), torch.flatten(Zcomp))
            Z_grad = torch.autograd.grad(loss_grad.real, paramdiff, allow_unused=True)[0]
            gradient += -2*np.real(Z_grad.detach().numpy())

            # Ey
            print("Calc Ey gradient:"+datetime.now().isoformat())
            sx, sy, sz = 0.0*self.sim._dx, 0.5*self.sim._dy, -0.5*self.sim._dz
            z = torch.linspace(sz+l_inds[0]*self.sim.dz, sz+(l_inds[0]+sizes[0])*self.sim.dz, sizes[0])
            shape = self.update_system_diffgeo(paramdiff, sx=sx, sy=sy)
            Z = shape.unsqueeze(-1).expand(-1,-1,z.shape[0])
            # Z = Z * torch.sigmoid(ksigz*(z-zmin))*torch.sigmoid(-ksigz*(z-zmax)).unsqueeze(0).unsqueeze(0).expand(shape.shape[0], shape.shape[1], -1)
            # Z += torch.sigmoid(-ksigz*(z-zmin)).unsqueeze(0).unsqueeze(0).expand(shape.shape[0], shape.shape[1], -1)
            Z = Z * torch.clamp((z-zmin)/sim._dz+0.5, min=0.0, max=1.0)*torch.clamp(-(z-zmax)/sim._dz+0.5, min=0.0, max=1.0).unsqueeze(0).unsqueeze(0).expand(shape.shape[0], shape.shape[1], -1)
            Z += torch.clamp(-(z-zmin)/sim._dz+0.5, min=0.0, max=1.0).unsqueeze(0).unsqueeze(0).expand(shape.shape[0], shape.shape[1], -1)

            Z = self.sim.perturb_epsclad + (self.sim.perturb_epscore-self.sim.perturb_epsclad) * Z
            Zcomp = torch.complex(Z, torch.zeros_like(Z)).permute(2,1,0)
            fields_tensor = torch.tensor(1j*self.sim._Ey_fwd_t0_pbox[...]*self.sim._Ey_adj_t0_pbox[...])
            loss_grad = torch.dot(torch.flatten(fields_tensor), torch.flatten(Zcomp))
            Z_grad = torch.autograd.grad(loss_grad.real, paramdiff, allow_unused=True)[0]
            gradient += -2*np.real(Z_grad.detach().numpy())

            # Ez
            print("Calc Ez gradient:"+datetime.now().isoformat())
            sx, sy, sz = 0.0*self.sim._dx, 0.0*self.sim._dy, 0.0*self.sim._dz
            z = torch.linspace(sz+l_inds[0]*self.sim.dz, sz+(l_inds[0]+sizes[0])*self.sim.dz, sizes[0])
            shape = self.update_system_diffgeo(paramdiff, sx=sx, sy=sy)
            Z = shape.unsqueeze(-1).expand(-1,-1,z.shape[0])
            # Z = Z * torch.sigmoid(ksigz*(z-zmin))*torch.sigmoid(-ksigz*(z-zmax)).unsqueeze(0).unsqueeze(0).expand(shape.shape[0], shape.shape[1], -1)
            # Z += torch.sigmoid(-ksigz*(z-zmin)).unsqueeze(0).unsqueeze(0).expand(shape.shape[0], shape.shape[1], -1)
            Z = Z * torch.clamp((z-zmin)/sim._dz+0.5, min=0.0, max=1.0)*torch.clamp(-(z-zmax)/sim._dz+0.5, min=0.0, max=1.0).unsqueeze(0).unsqueeze(0).expand(shape.shape[0], shape.shape[1], -1)
            Z += torch.clamp(-(z-zmin)/sim._dz+0.5, min=0.0, max=1.0).unsqueeze(0).unsqueeze(0).expand(shape.shape[0], shape.shape[1], -1)

            Z = self.sim.perturb_epsclad + (self.sim.perturb_epscore-self.sim.perturb_epsclad) * Z
            Zcomp = torch.complex(Z, torch.zeros_like(Z)).permute(2,1,0)
            fields_tensor = torch.tensor(1j*self.sim._Ez_fwd_t0_pbox[...]*self.sim._Ez_adj_t0_pbox[...])
            loss_grad = torch.dot(torch.flatten(fields_tensor), torch.flatten(Zcomp))
            Z_grad = torch.autograd.grad(loss_grad.real, paramdiff, allow_unused=True)[0]
            gradient += -2*np.real(Z_grad.detach().numpy())

            return gradient
        else:
            update_boxes = self.get_update_boxes(sim, params)
            bbox = update_boxes[0]
            bbox = DomainCoordinates(bbox[0], bbox[1], bbox[2], 
                                     bbox[3], bbox[4], bbox[5], 
                                     sim._dx, sim._dy, sim._dz)
            g_inds, l_inds, d_inds, sizes = sim._FDTD__get_local_domain_overlap(bbox)

            ### unperturbed eps
            sim._eps_x_p0 = sim._dap.createGlobalVec()
            sim._eps_y_p0 = sim._dap.createGlobalVec()
            sim._eps_z_p0 = sim._dap.createGlobalVec()
            ### perturbed eps
            sim._eps_x_p = sim._dap.createGlobalVec()
            sim._eps_y_p = sim._dap.createGlobalVec()
            sim._eps_z_p = sim._dap.createGlobalVec()
            i0,j0,k0 = g_inds
            I,J,K = sizes
            sim._eps.get_values(k0,k0+K,j0,j0+J,i0,i0+I,
                                sx=0.5, sy=0.0, sz=-0.5,
                                arr=sim._eps_x_p0.getArray())
            sim._eps.get_values(k0,k0+K,j0,j0+J,i0,i0+I,
                                sx=0.0, sy=0.5, sz=-0.5,
                                arr=sim._eps_y_p0.getArray())
            sim._eps.get_values(k0,k0+K,j0,j0+J,i0,i0+I,
                                sx=0.0, sy=0.0, sz=0.0,
                                arr=sim._eps_z_p0.getArray())

            Ai = sim.get_A_diag()

            step = self._step
            lenp = len(params)

            grad_full = None
            grad_parts = []
            if(RANK == 0):
                grad_full = np.zeros(N_PROC, dtype=np.double)

            gradient = np.zeros(lenp)
            for i in range(lenp):
                print("param",i+1,"of",lenp, datetime.now().isoformat()+'\033[1A\r')
                p0 = params[i]
                ub = update_boxes[i]

                # perturb the system
                params[i] += step
                self.update_system(params)

                if(type(ub[0]) == list or type(ub[0]) == np.ndarray or \
                   type(ub[0]) == tuple):
                    for box in ub:
                        self.sim.perturb(box)
                else:
                    self.sim.perturb(ub)

                # calculate dAdp and assemble the full result on the master node
                product = sim.calc_ydAx(Ai)

                # ### debug field and eps
                # eps_x,eps_y,eps_z=Ai
                # nslice=sim._eps_x_p.getArray().real[K*J*7:K*J*8]
                # nidx=np.linspace(0,K*J-1,K*J)
                # nx=np.mod(nidx,K)
                # ny=np.floor(nidx/K)
                # np.savetxt("eps_xp.csv",np.transpose([eps_x.real[K*J*7:K*J*8],nslice,nx,ny]),fmt='%.18e',delimiter=',',header='n0,np,x,y',comments='')
                # np.savetxt("fields.csv",np.transpose([np.abs(sim._Ey_fwd_t0_pbox[K*J*7:K*J*8]),np.abs(sim._Ey_adj_t0_pbox[K*J*7:K*J*8]),nx,ny]),fmt='%.18e',delimiter=',',header='fwd,adj,x,y',comments='')
                # input("Press Enter to continue")

                grad_part = -2*np.real( product/step )
                grad_parts.append(grad_part)

                # # revert the system to its original state
                params[i] = p0
                # self.update_system(params)
                # if(type(ub[0]) == list or type(ub[0]) == np.ndarray or \
                #    type(ub[0]) == tuple):
                #     for box in ub:
                #         self.sim.update(box)
                # else:
                #     self.sim.update(ub)

            COMM.Barrier()
            for i in range(lenp):
                # send the partially computed gradient to the master node to finish
                # up the calculation
                #COMM.Gather(grad_parts[i], grad_full, root=0)
                grad_full = COMM.gather(grad_parts[i], root=0)

                # finish calculating the gradient
                if(NOT_PARALLEL):
                    gradient[i] = np.sum(grad_full)

            sim._eps_x_p0.destroy()
            sim._eps_y_p0.destroy()
            sim._eps_z_p0.destroy()
            del sim._eps_x_p0; del sim._eps_y_p0; del sim._eps_z_p0
            sim._eps_x_p.destroy()
            sim._eps_y_p.destroy()
            sim._eps_z_p.destroy()
            del sim._eps_x_p; del sim._eps_y_p; del sim._eps_z_p

            if(NOT_PARALLEL):
                return gradient

    def gradient(self, params):
        """Manage the calculation of the gradient figure of merit.

        To calculate the gradient, we update the system, run a forward and
        adjoint simulation, and then calculate the gradient using
        :func:`calc_gradient`.  Most of these operations are done in parallel
        using MPI.

        Parameters
        ----------
        params : numpy.ndarray
            List of design parameters of the system

        Returns
        -------
        numpy.ndarray
            (Master node only) The gradient of the figure of merit computed  with 
            respect to the design variables
        """
        # update system
        print('Update system:'+datetime.now().isoformat())
        self.update_system(params)
        self.sim.update()

        # run the forward simulation
        if(not np.array_equal(self.prev_params, params)):
            self.sim.solve_forward()

        # get dFdx which will be used as the source for the adjoint simulation
        # dFdx is calculated on the root node and then broadcast to the other
        # nodes.
        # TODO: parallelize this operation?
        comm = MPI.COMM_WORLD

        # This should return only non-null on RANK=0
        print('Calc dFdx:'+datetime.now().isoformat())
        dFdx = self.calc_dFdx(self.sim, params)

        # Reduce memory footprint of sim._E_fwd_t0 after dFdx calculation
        #self.sim._Hx_fwd_t0.destroy()
        #self.sim._Hy_fwd_t0.destroy()
        #self.sim._Hz_fwd_t0.destroy()
        #del self.sim._Hx_fwd_t0
        #del self.sim._Hy_fwd_t0
        #del self.sim._Hz_fwd_t0

        self.sim._Ex_fwd_t0_pbox = self.sim._dap.createGlobalVec()
        self.sim._Ey_fwd_t0_pbox = self.sim._dap.createGlobalVec()
        self.sim._Ez_fwd_t0_pbox = self.sim._dap.createGlobalVec()
        self.sim.get_pbox_field_fwd()

        #self.sim._Ex_fwd_t0.destroy()
        #self.sim._Ey_fwd_t0.destroy()
        #self.sim._Ez_fwd_t0.destroy()
        #del self.sim._Ex_fwd_t0
        #del self.sim._Ey_fwd_t0
        #del self.sim._Ez_fwd_t0

        #if(isinstance(self.sim, fdfd.FDFD_TE)):
        #dFdx = comm.bcast(dFdx, root=0)
        #elif(isinstance(self.sim, fdfd.FDFD_3D)):
        #    pass

        # run the adjoint source
        print('Set adj src:'+datetime.now().isoformat())
        self.sim.set_adjoint_sources(dFdx)
        self.sim.solve_adjoint()

        # Reduce memory footprint of sim._E_adj_t0 after dFdx calculation
        self.sim._Hx_adj_t0.destroy()
        self.sim._Hy_adj_t0.destroy()
        self.sim._Hz_adj_t0.destroy()
        del self.sim._Hx_adj_t0; del self.sim._Hy_adj_t0; del self.sim._Hz_adj_t0

        self.sim._Ex_adj_t0_pbox = self.sim._dap.createGlobalVec()
        self.sim._Ey_adj_t0_pbox = self.sim._dap.createGlobalVec()
        self.sim._Ez_adj_t0_pbox = self.sim._dap.createGlobalVec()
        self.sim.get_pbox_field_adj()

        self.sim._Ex_adj_t0.destroy()
        self.sim._Ey_adj_t0.destroy()
        self.sim._Ez_adj_t0.destroy()
        del self.sim._Ex_adj_t0; del self.sim._Ey_adj_t0; del self.sim._Ez_adj_t0

        if(NOT_PARALLEL):
            info_message('Calculating gradient...')

        grad_f = self.calc_gradient(self.sim, params)
        grad_p = self.calc_grad_p(self.sim, params)

        self.sim._Ex_fwd_t0_pbox.destroy()
        self.sim._Ey_fwd_t0_pbox.destroy()
        self.sim._Ez_fwd_t0_pbox.destroy()
        self.sim._Ex_adj_t0_pbox.destroy()
        self.sim._Ex_adj_t0_pbox.destroy()
        self.sim._Ex_adj_t0_pbox.destroy()
        # del self.sim._Ex_fwd_t0_pbox; del self.sim._Ey_fwd_t0_pbox; del self.sim._Ez_fwd_t0_pbox
        # del self.sim._Ex_adj_t0_pbox; del self.sim._Ey_adj_t0_pbox; del self.sim._Ez_adj_t0_pbox

        if(NOT_PARALLEL):
            self.current_gradient = grad_f
            return grad_f + grad_p
        else:
            return None

    def check_gradient(self, params, indices=[], plot=True, verbose=True,
                       return_gradients=False, fd_step=1e-10):
        """Verify that the gradient is accurate.

        It is highly recommended that the accuracy of the gradients be checked
        prior to being used. If the accuracy is above ~1%, it is likely that
        there is an inconsistency between how the figure of merit and adjoint
        sources (dFdx) are being computed.

        The adjoint method gradient error is evaluated by comparing the
        gradient computed using the adjoint method to a gradient computed by
        direct finite differences.  In other words, the "correct" derivatives
        to which the adjoint method gradient is compared are given by

        .. math::
            \\frac{\partial F}{\partial p_i} \\approx \\frac{F(p_i + \Delta p) - F(p_i)}{\Delta p}

        Note that this method for calculating the gradient is not used in a
        practical setting because it requires performing N+1 simulations in
        order to compute the gradient with respect to N design variables
        (compared to only 2 simulations in the case of the adjoint method).

        Parameters
        ----------
        params : numpy.ndarray
            design parameters
        indices : list or numpy.ndarray
            list of gradient indices to check. An empty list indicates that the
            whole gradient should be verified. A subset of indices may be
            desirable for large problems.  (default = [])
        plot : bool (optional)
            Plot the gradients and errors (default = True)
        verbose : bool (optional)
            If True, print progress (default = True)
        return_gradients : bool
            If True, return the gradient arrays (default = False)

        Returns
        -------
        float
            Relative error in gradient.
        """

        if(indices == []):
            indices = np.arange(0, len(params),1)

        # make sure everything is up to date
        self.update_system(params)
        self.sim.update()

        grad_am = self.gradient(params)
        grad_fd = np.zeros(len(indices))

        fom0 = self.fom(params)

        # calculate the "true" derivatives using finite differences
        if(NOT_PARALLEL and verbose):
            info_message('Checking gradient...')

        for i in range(len(indices)):
            if(NOT_PARALLEL and verbose):
                print('\tDerivative %d of %d' % (i+1, len(indices)))

            j = indices[i]
            p0 = params[j]
            params[j] += fd_step
            fom1 = self.fom(params)
            if(NOT_PARALLEL):
                grad_fd[i] = (fom1-fom0)/fd_step
            params[j] = p0

        if(NOT_PARALLEL):
            errors = np.abs(grad_fd - grad_am[indices]) / np.abs(grad_fd)
            error_tot = np.linalg.norm(grad_fd - grad_am[indices]) / np.linalg.norm(grad_fd)

            if(error_tot < 0.01 and verbose):
                info_message('The total error in the gradient is %0.4E' % \
                             (error_tot))
            else:
                warning_message('The total error in the gradient is %0.4E '
                                'which is over 1%%' % (error_tot), \
                                'emopt.adjoint_method')

            if(plot):
                import matplotlib.pyplot as plt
                f = plt.figure()
                ax1 = f.add_subplot(311)
                ax2 = f.add_subplot(312)
                ax3 = f.add_subplot(313)

                xs = np.arange(len(indices))
                ax1.bar(xs, grad_fd)
                ax1.set_title('Finite Differences')
                ax2.bar(xs, grad_am[indices])
                ax2.set_title('Adjoint Method')
                ax3.bar(xs, errors)
                ax3.set_title('Error in Adjoint Method')

                for ax in [ax1, ax2, ax3]:
                    ax.set_xticklabels(['%d' % i for i in indices])

                ax3.set_yscale('log', nonposy='clip')

                plt.savefig('Check_gradient.png', dpi=600)
                plt.close()

            if(return_gradients):
                return error_tot, grad_fd, grad_am
            else:
                return error_tot
        else:
            if(return_gradients):
                return None, None, None
            return None

class AdjointMethodMORE(with_metaclass(ABCMeta, AdjointMethod)):
    """An AdjointMethod object for an ensemble of different figures of merit
    (Multi-objective adjoint method) - reuse perturbed eps for gradient
    calculations of all FOMs in one pass

    In many situations, it is desirable to calculate the sensitivities of a
    structure corresponding to multiple objective functions.  A simple common
    exmaple of this a broadband figure of merit which considers the performance
    of structure at a range of different excitation frequencies/wavelengths.
    In other cases, it may be desirable to calculate a total sensitivity which
    is made up of two different figures of merit which are calculated for the
    same excitation.

    In either case, we need a way to easily handle these more complicated
    figures of merits and their gradients (i.e. the sensitivities). This class
    provides a simple interface to do just that.  By overriding calc_total_fom
    and calc_total_gradient, you can build up more complicated figures of
    merit.

    Parameters
    ----------
    ams : list of :class:`.AdjointMethod`
        A list containing *extended* AdjointMethod objects

    Attributes
    ----------
    adjoint_methods : list of :class:`.AdjointMethod`
        A list containing extended AdjointMethod objects
    """

    def __init__(self, ams, step=1e-6):
        self._ams = ams
        self._foms_current = np.zeros(len(ams))
        self._step = step

    @property
    def adjoint_methods(self):
        return self._ams

    @adjoint_methods.setter
    def adjoint_methods(self, new_ams):
        self._ams = new_ams

    def update_system(self, params):
        """Update all of the individual AdjointMethods."""
        for am in self._ams:
            am.update_system(params)

    def calc_fom(self, sim, params):
        """Calculate the figure of merit.
        """
        # this just redirects to calc_total_foms
        return self.calc_total_fom(self._foms_current)

    def calc_dFdx(self, sim, params):
        # We dont need this -- all dFdx's are performed by
        # AdjointMethod objects contained in self._ams
        pass

    def calc_grad_p(self, sim, params):
        # We dont need this -- all individual grad_p calculations are handled
        # by supplied AdjointMethod objects.
        pass

    @abstractmethod
    def calc_total_fom(self, foms):
        """Calculate the 'total' figure of merit based on a list of evaluated
        objective functions.

        The user should override this function in order to define how all of the
        individual figures of merit are combined to form a single 'total'
        figure of merit. This may be a sum of the input FOMs, a minimax of the
        FOMs, etc.  A common example is to combine figures of merit calculated
        for different wavelengths of operation.

        See Also
        --------
        :ref:`emopt.fomutils` : functions which may be useful for combining figures of merit

        Parameters
        ----------
        foms : list of float
            List containing individual FOMs which are used to compute the total
            figure of merit.

        Returns
        -------
        float
            The total figure of merit
        """
        pass

    @abstractmethod
    def calc_total_gradient(self, foms, grads):
        """Calculate the 'total' gradient of a figure of merit based on a list
        of evaluated objective functions.

        The user should override this function in order to define the gradient
        of the total figure of merit.

        See Also
        --------
        :ref:`emopt.fomutils` : functions which may be useful for combining figures of merit and their gradients.

        Parameters
        ----------
        foms : list
            List of individual foms
        grads : list
            List of individual grads

        Returns
        -------
        numpy.ndarray
            1D numpy array containing total gradient. note: the output vector
            should have the same shape as the input vectors contained in grads
        """
        pass

    def fom(self, params):
        """Calculate the total figure of merit.

        Notes
        -----
        Overrides :class:`.AdjointMethod`.fom(...)

        Parameters
        ----------
        params : numpy.ndarray
            Design parameters

        Returns
        -------
        float
            (Master node only) The total figure of merit
        """
        foms = []

        for am in self._ams:
            foms.append(am.fom(params))
            # del am.sim._Ex; del am.sim._Ey; del am.sim._Ez
            # del am.sim._Hx; del am.sim._Hy; del am.sim._Hz
        gc.collect()

        self._foms_current = foms
        if(NOT_PARALLEL):
            fom_total = self.calc_total_fom(foms)
            return fom_total
        else:
            return None

    def gradient(self, params):
        """Calculate the total gradient.

        Notes
        -----
        Overrides :class:`.AdjointMethod`.gradient(...)

        Parameters
        ----------
        params : numpy.ndarray
            Design parameters with respect to which gradient is evaluated

        Returns
        -------
        numpy.ndarray
            (Master node only) The gradient of total figure of merit.
        """
        foms = []
        grads = []

        gc.collect()

        for am in self._ams:
            am.update_system(params)
            am.sim.update()
            if(not np.array_equal(am.prev_params, params)):
                am.sim.solve_forward()

            # comm = MPI.COMM_WORLD
            dFdx = am.calc_dFdx(am.sim, params)
            # dFdx = comm.bcast(dFdx, root=0)

            # Reduce memory footprint after dFdx calculation
            am.sim._Hx_fwd_t0.destroy()
            am.sim._Hy_fwd_t0.destroy()
            am.sim._Hz_fwd_t0.destroy()

            am.sim._Ex_fwd_t0_pbox = am.sim._dap.createGlobalVec()
            am.sim._Ey_fwd_t0_pbox = am.sim._dap.createGlobalVec()
            am.sim._Ez_fwd_t0_pbox = am.sim._dap.createGlobalVec()
            am.sim.get_pbox_field_fwd()

            am.sim._Ex_fwd_t0.destroy()
            am.sim._Ey_fwd_t0.destroy()
            am.sim._Ez_fwd_t0.destroy()

            am.sim.set_adjoint_sources(dFdx)
            am.sim.solve_adjoint()

            # Reduce memory footprint after adjoint simulation
            am.sim._Hx_adj_t0.destroy()
            am.sim._Hy_adj_t0.destroy()
            am.sim._Hz_adj_t0.destroy()

            am.sim._Ex_adj_t0_pbox = am.sim._dap.createGlobalVec()
            am.sim._Ey_adj_t0_pbox = am.sim._dap.createGlobalVec()
            am.sim._Ez_adj_t0_pbox = am.sim._dap.createGlobalVec()
            am.sim.get_pbox_field_adj()

            am.sim._Ex_adj_t0.destroy()
            am.sim._Ey_adj_t0.destroy()
            am.sim._Ez_adj_t0.destroy()

            foms.append( am.calc_fom(am.sim, params) )
            del am.sim._Ex_fwd_t0; del am.sim._Ey_fwd_t0; del am.sim._Ez_fwd_t0
            del am.sim._Hx_fwd_t0; del am.sim._Hy_fwd_t0; del am.sim._Hz_fwd_t0
            del am.sim._Ex_adj_t0; del am.sim._Ey_adj_t0; del am.sim._Ez_adj_t0
            del am.sim._Hx_adj_t0; del am.sim._Hy_adj_t0; del am.sim._Hz_adj_t0

        # for am in self._ams:
        #     del am.sim._Ex; del am.sim._Ey; del am.sim._Ez;
        #     del am.sim._Hx; del am.sim._Hy; del am.sim._Hz;
        gc.collect()

        pos,lens = self._ams[0].sim._da.getCorners()
        k0,j0,i0 = pos
        K,J,I = lens
        for i in range(len(self._ams)):
            if i==0:
                self._ams[i].sim._eps_x_p = self._ams[i].sim._da.createGlobalVec()
                self._ams[i].sim._eps_y_p = self._ams[i].sim._da.createGlobalVec()
                self._ams[i].sim._eps_z_p = self._ams[i].sim._da.createGlobalVec()
                self._ams[i].sim._eps.get_values(k0,k0+K,j0,j0+J,i0,i0+I,
                                                 sx=0.5, sy=0.0, sz=-0.5,
                                                 arr=self._ams[i].sim._eps_x_p.getArray())
                self._ams[i].sim._eps.get_values(k0,k0+K,j0,j0+J,i0,i0+I,
                                                 sx=0.0, sy=0.5, sz=-0.5,
                                                 arr=self._ams[i].sim._eps_y_p.getArray())
                self._ams[i].sim._eps.get_values(k0,k0+K,j0,j0+J,i0,i0+I,
                                                 sx=0.0, sy=0.0, sz=0.0,
                                                 arr=self._ams[i].sim._eps_z_p.getArray())

            else:
                self._ams[i].sim._eps_x_p = self._ams[0].sim._eps_x_p
                self._ams[i].sim._eps_y_p = self._ams[0].sim._eps_y_p
                self._ams[i].sim._eps_z_p = self._ams[0].sim._eps_z_p

        Ai = self._ams[0].sim.get_A_diag()
        step = self._ams[0]._step
        update_boxes = self._ams[0].get_update_boxes(self._ams[0].sim, params)
        lenp = len(params)

        if (NOT_PARALLEL):
            info_message('Calculating gradient...')

        gradient = np.zeros((len(self._ams), lenp))
        # AdjointMethodMORE is to be used only on single node, so grad_parts == gradient
        # need to add COMM.Barrier/COMM.gather for parallelized version
        for i in range(lenp):
            print("param",i+1,"of",lenp,'\033[1A\r')
            p0 = params[i]
            ub = update_boxes[i]

            # perturb the system
            params[i] += step
            self._ams[0].update_system(params)
            if(type(ub[0]) == list or type(ub[0]) == np.ndarray or \
                type(ub[0]) == tuple):
                for box in ub:
                    self._ams[0].sim.perturb(box)
            else:
                self._ams[0].sim.perturb(ub)

            # calculate dAdp and assemble the full result on the master node
            for j in range(len(self._ams)):
                # clone perturbed eps to all other AM objects than the head
                if j!=0:
                    self._ams[j].sim._i0 = self._ams[0].sim._i0
                    self._ams[j].sim._i1 = self._ams[0].sim._i1
                gradient[j,i] = -2*np.real( self._ams[j].sim.calc_ydAx(Ai)/step )

            # revert the system to its original state
            params[i] = p0

        for i in range(len(self._ams)):
            self._ams[i].current_gradient = gradient[i]

        for am in self._ams:
            am.sim._eps_x_p.destroy()
            am.sim._eps_y_p.destroy()
            am.sim._eps_z_p.destroy()
            del am.sim._eps_x_p
            del am.sim._eps_y_p
            del am.sim._eps_z_p

            am.sim._Ex_fwd_t0_pbox.destroy()
            am.sim._Ey_fwd_t0_pbox.destroy()
            am.sim._Ez_fwd_t0_pbox.destroy()
            am.sim._Ex_adj_t0_pbox.destroy()
            am.sim._Ey_adj_t0_pbox.destroy()
            am.sim._Ez_adj_t0_pbox.destroy()

        if(NOT_PARALLEL):
            grad_total = self.calc_total_gradient(foms, gradient)
            return grad_total
        else:
            return None


    def check_gradient(self, params, indices=[], plot=True, verbose=True,
                       return_gradients=False, fd_step=1e-10):
        """Check the gradient of an multi-objective AdjointMethod.

        Parameters
        ----------
        params : numpy.ndarray
            design parameters
        indices : list or numpy.ndarray
            list of gradient indices to check. An empty list indicates that the
            whole gradient should be verified. A subset of indices may be
            desirable for large problems.  (default = [])

        Returns
        -------
        float
            Relative error in gradient.
        """
        # we override this function so we can initially update all of the ams
        # as desired
        self.sim = self._ams[0].sim
        for am in self._ams:
            am.update_system(params)
            am.sim.update()

        return super(AdjointMethodMO, self).check_gradient(params, indices, plot,
                                                           verbose,
                                                           return_gradients,
                                                           fd_step)

class AdjointMethodMOSE(with_metaclass(ABCMeta, AdjointMethod)):
    """An AdjointMethod object for an ensemble of different figures of merit
    (Multi-objective adjoint method) - sequential execution of multiple objectives
    that can reuse the same geometry

    In many situations, it is desirable to calculate the sensitivities of a
    structure corresponding to multiple objective functions.  A simple common
    exmaple of this a broadband figure of merit which considers the performance
    of structure at a range of different excitation frequencies/wavelengths.
    In other cases, it may be desirable to calculate a total sensitivity which
    is made up of two different figures of merit which are calculated for the
    same excitation.

    In either case, we need a way to easily handle these more complicated
    figures of merits and their gradients (i.e. the sensitivities). This class
    provides a simple interface to do just that.  By overriding calc_total_fom
    and calc_total_gradient, you can build up more complicated figures of
    merit.

    Parameters
    ----------
    ams : list of :class:`.AdjointMethod`
        A list containing *extended* AdjointMethod objects

    Attributes
    ----------
    adjoint_methods : list of :class:`.AdjointMethod`
        A list containing extended AdjointMethod objects
    """

    def __init__(self, ams, step=1e-6):
        self._ams = ams
        self._foms_current = np.zeros(len(ams))
        self._step = step
        self._grads = []

    @property
    def adjoint_methods(self):
        return self._ams

    @adjoint_methods.setter
    def adjoint_methods(self, new_ams):
        self._ams = new_ams

    def update_system(self, params):
        """Update all of the individual AdjointMethods."""
        for am in self._ams:
            am.update_system(params)

    def calc_fom(self, sim, params):
        """Calculate the figure of merit.
        """
        # this just redirects to calc_total_foms
        return self.calc_total_fom(self._foms_current)

    def calc_dFdx(self, sim, params):
        # We dont need this -- all dFdx's are performed by
        # AdjointMethod objects contained in self._ams
        pass

    def calc_grad_p(self, sim, params):
        # We dont need this -- all individual grad_p calculations are handled
        # by supplied AdjointMethod objects.
        pass

    @abstractmethod
    def calc_total_fom(self, foms):
        """Calculate the 'total' figure of merit based on a list of evaluated
        objective functions.

        The user should override this function in order to define how all of the
        individual figures of merit are combined to form a single 'total'
        figure of merit. This may be a sum of the input FOMs, a minimax of the
        FOMs, etc.  A common example is to combine figures of merit calculated
        for different wavelengths of operation.

        See Also
        --------
        :ref:`emopt.fomutils` : functions which may be useful for combining figures of merit

        Parameters
        ----------
        foms : list of float
            List containing individual FOMs which are used to compute the total
            figure of merit.

        Returns
        -------
        float
            The total figure of merit
        """
        pass

    @abstractmethod
    def calc_total_gradient(self, foms, grads):
        """Calculate the 'total' gradient of a figure of merit based on a list
        of evaluated objective functions.

        The user should override this function in order to define the gradient
        of the total figure of merit.

        See Also
        --------
        :ref:`emopt.fomutils` : functions which may be useful for combining figures of merit and their gradients.

        Parameters
        ----------
        foms : list
            List of individual foms
        grads : list
            List of individual grads

        Returns
        -------
        numpy.ndarray
            1D numpy array containing total gradient. note: the output vector
            should have the same shape as the input vectors contained in grads
        """
        pass

    def fom(self, params):
        """Calculate the total figure of merit.

        Notes
        -----
        Overrides :class:`.AdjointMethod`.fom(...)

        Parameters
        ----------
        params : numpy.ndarray
            Design parameters

        Returns
        -------
        float
            (Master node only) The total figure of merit
        """
        foms = []
        self._grads = []
        for am in self._ams:
            foms.append(am.fom(params))
            self._grads.append( am.gradient(params))

        self._foms_current = foms
        if(NOT_PARALLEL):
            fom_total = self.calc_total_fom(foms)
            return fom_total
        else:
            return None

    def gradient(self, params):
        """Calculate the total gradient.

        Notes
        -----
        Overrides :class:`.AdjointMethod`.gradient(...)

        Parameters
        ----------
        params : numpy.ndarray
            Design parameters with respect to which gradient is evaluated

        Returns
        -------
        numpy.ndarray
            (Master node only) The gradient of total figure of merit.
        """
        foms = []
        # grads = []

        grads = self._grads
        for am in self._ams:
            # grads.append( am.gradient(params) )
            foms.append( am.calc_fom(am.sim, params) )

        if(NOT_PARALLEL):
            grad_total = self.calc_total_gradient(foms, grads)
            return grad_total
        else:
            return None


    def check_gradient(self, params, indices=[], plot=True, verbose=True,
                       return_gradients=False, fd_step=1e-10):
        """Check the gradient of an multi-objective AdjointMethod.

        Parameters
        ----------
        params : numpy.ndarray
            design parameters
        indices : list or numpy.ndarray
            list of gradient indices to check. An empty list indicates that the
            whole gradient should be verified. A subset of indices may be
            desirable for large problems.  (default = [])

        Returns
        -------
        float
            Relative error in gradient.
        """
        # we override this function so we can initially update all of the ams
        # as desired
        self.sim = self._ams[0].sim
        for am in self._ams:
            am.update_system(params)
            am.sim.update()

        return super(AdjointMethodMO, self).check_gradient(params, indices, plot,
                                                           verbose,
                                                           return_gradients,
                                                           fd_step)

class AdjointMethodMO(with_metaclass(ABCMeta, AdjointMethod)):
    """An AdjointMethod object for an ensemble of different figures of merit
    (Multi-objective adjoint method).

    In many situations, it is desirable to calculate the sensitivities of a
    structure corresponding to multiple objective functions.  A simple common
    exmaple of this a broadband figure of merit which considers the performance
    of structure at a range of different excitation frequencies/wavelengths.
    In other cases, it may be desirable to calculate a total sensitivity which
    is made up of two different figures of merit which are calculated for the
    same excitation.

    In either case, we need a way to easily handle these more complicated
    figures of merits and their gradients (i.e. the sensitivities). This class
    provides a simple interface to do just that.  By overriding calc_total_fom
    and calc_total_gradient, you can build up more complicated figures of
    merit.

    Parameters
    ----------
    ams : list of :class:`.AdjointMethod`
        A list containing *extended* AdjointMethod objects

    Attributes
    ----------
    adjoint_methods : list of :class:`.AdjointMethod`
        A list containing extended AdjointMethod objects
    """

    def __init__(self, ams, step=1e-6):
        self._ams = ams
        self._foms_current = np.zeros(len(ams))
        self._step = step

    @property
    def adjoint_methods(self):
        return self._ams

    @adjoint_methods.setter
    def adjoint_methods(self, new_ams):
        self._ams = new_ams

    def update_system(self, params):
        """Update all of the individual AdjointMethods."""
        for am in self._ams:
            am.update_system(params)

    def calc_fom(self, sim, params):
        """Calculate the figure of merit.
        """
        # this just redirects to calc_total_foms
        return self.calc_total_fom(self._foms_current)

    def calc_dFdx(self, sim, params):
        # We dont need this -- all dFdx's are performed by
        # AdjointMethod objects contained in self._ams
        pass

    def calc_grad_p(self, sim, params):
        # We dont need this -- all individual grad_p calculations are handled
        # by supplied AdjointMethod objects.
        pass

    @abstractmethod
    def calc_total_fom(self, foms):
        """Calculate the 'total' figure of merit based on a list of evaluated
        objective functions.

        The user should override this function in order to define how all of the
        individual figures of merit are combined to form a single 'total'
        figure of merit. This may be a sum of the input FOMs, a minimax of the
        FOMs, etc.  A common example is to combine figures of merit calculated
        for different wavelengths of operation.

        See Also
        --------
        :ref:`emopt.fomutils` : functions which may be useful for combining figures of merit

        Parameters
        ----------
        foms : list of float
            List containing individual FOMs which are used to compute the total
            figure of merit.

        Returns
        -------
        float
            The total figure of merit
        """
        pass

    @abstractmethod
    def calc_total_gradient(self, foms, grads):
        """Calculate the 'total' gradient of a figure of merit based on a list
        of evaluated objective functions.

        The user should override this function in order to define the gradient
        of the total figure of merit.

        See Also
        --------
        :ref:`emopt.fomutils` : functions which may be useful for combining figures of merit and their gradients.

        Parameters
        ----------
        foms : list
            List of individual foms
        grads : list
            List of individual grads

        Returns
        -------
        numpy.ndarray
            1D numpy array containing total gradient. note: the output vector
            should have the same shape as the input vectors contained in grads
        """
        pass

    def fom(self, params):
        """Calculate the total figure of merit.

        Notes
        -----
        Overrides :class:`.AdjointMethod`.fom(...)

        Parameters
        ----------
        params : numpy.ndarray
            Design parameters

        Returns
        -------
        float
            (Master node only) The total figure of merit
        """
        foms = []

        for am in self._ams:
            foms.append(am.fom(params))

        self._foms_current = foms
        if(NOT_PARALLEL):
            fom_total = self.calc_total_fom(foms)
            return fom_total
        else:
            return None

    def gradient(self, params):
        """Calculate the total gradient.

        Notes
        -----
        Overrides :class:`.AdjointMethod`.gradient(...)

        Parameters
        ----------
        params : numpy.ndarray
            Design parameters with respect to which gradient is evaluated

        Returns
        -------
        numpy.ndarray
            (Master node only) The gradient of total figure of merit.
        """
        foms = []
        grads = []

        for am in self._ams:
            grads.append( am.gradient(params) )
            foms.append( am.calc_fom(am.sim, params) )

        if(NOT_PARALLEL):
            grad_total = self.calc_total_gradient(foms, grads)
            return grad_total
        else:
            return None

    def calc_gradient_manual(self, sim, params):
        pass

    def check_gradient(self, params, indices=[], plot=True, verbose=True,
                       return_gradients=False, fd_step=1e-10):
        """Check the gradient of an multi-objective AdjointMethod.

        Parameters
        ----------
        params : numpy.ndarray
            design parameters
        indices : list or numpy.ndarray
            list of gradient indices to check. An empty list indicates that the
            whole gradient should be verified. A subset of indices may be
            desirable for large problems.  (default = [])

        Returns
        -------
        float
            Relative error in gradient.
        """
        # we override this function so we can initially update all of the ams
        # as desired
        self.sim = self._ams[0].sim
        for am in self._ams:
            am.update_system(params)
            am.sim.update()

        return super(AdjointMethodMO, self).check_gradient(params, indices, plot,
                                                           verbose,
                                                           return_gradients,
                                                           fd_step)

class AdjointMethodFM2D(AdjointMethod):
    """Define an :class:`.AdjointMethod` which simplifies the calculation of
    gradients which are a function of the materials (eps and mu) in 2D
    problems.

    In certain cases, the gradient of a function of the fields, permittivity,
    and permeability may be desired.  Differentiating the function with respect
    to the permittivity and permeability shares many of the same calculations
    in common with :meth:`.AdjointMethod.calc_gradient`.  In order to maximize
    performance and simplify the implementation of material-dependent figures
    of merit, this class reimplements the :meth:`calc_gradient` function.

    Attributes
    ----------
    sim : :class:`emopt.fdfd.FDFD`
        The simulation object
    step : float (optional)
        The step size used by gradient calculation (default = False)
    """
    def __init__(self, sim, step=1e-8):
        super(AdjointMethodFM2D, self).__init__(sim, step)

    @abstractmethod
    def calc_dFdm(self, sim, params):
        """Calculate the derivative of F with respect to :math:`\epsilon`,
        :math:`\epsilon^*`, :math:`\mu`, and :math:`\mu^*`

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        tuple of numpy.array
            The derivatives of F with respect to spatially-dependent eps and mu
            and their complex conjugates. These derivatives should be arrays
            with dimension (M,N) and should be returned in a tuple with the
            format (dFdeps, dFdeps_conf, dFdmu, dFdmu_conj)
        """
        pass

    def calc_gradient(self, sim, params):
        """Calculate the gradient of a figure of merit which depends on the
        permittivity and permeability.

        Parameters
        ----------
        sim : FDFD
            Simulation object.  sim = self.sim
        params : numpy.array or list of floats
            List of design parameters.

        Returns
        -------
        numpy.array
            **(Master node only)** Gradient of figure of merit, i.e. list of
            derivatives of fom with respect to each design variable
        """
        # Semantically, we would not normally need to override this method,
        # however it turns out that the operations needed to compute the
        # gradient of a field-dependent function and a permittivit-dependent
        # function are very similar (both require the calculation of the
        # derivative of the materials wrt the design parameters.)  For the sake
        # of performance, we combine the two calculations here.

        w_pml_l = sim.w_pml_left
        w_pml_r = sim.w_pml_right
        w_pml_t = sim.w_pml_top
        w_pml_b = sim.w_pml_bottom
        M = sim.M
        N = sim.N
        X = sim.X
        Y = sim.Y

        # get the current diagonal elements of A.
        # only these elements change when the design variables change.
        Af = PETSc.Vec()
        Ai = sim.get_A_diag()

        # Get the derivatives w.r.t. eps, mu
        if(NOT_PARALLEL):
            dFdeps, dFdeps_conj, dFdmu, dFdmu_conj = self.calc_dFdm(sim, params)

        step = self._step
        update_boxes = self.get_update_boxes(sim, params)
        lenp = len(params)

        grad_full = None
        if(RANK == 0):
            grad_full = np.zeros(sim.nunks, dtype=np.double)

        gradient = np.zeros(lenp)
        for i in range(lenp):
            p0 = params[i]
            ub = update_boxes[i]

            # perturb the system
            params[i] += step
            self.update_system(params)
            if(type(ub[0]) == list or type(ub[0]) == np.ndarray or \
               type(ub[0]) == tuple):
                for box in ub:
                    self.sim.update(box)
            else:
                self.sim.update(ub)

            # calculate derivative via y^T*dA/dp*x
            product = sim.calc_ydAx(Ai)
            grad_part = -2*np.real( product/step )

            # send the partially computed gradient to the master node to finish
            # up the calculation
            #MPI.COMM_WORLD.Gather(grad_part, grad_full, root=0)
            grad_full = MPI.COMM_WORLD.gather(grad_part, root=0)

            # We also need dAdp to account for the derivative of eps and mu
            # get the updated diagonal elements of A
            Af = sim.get_A_diag(Af)
            dAdp = (Af-Ai)/step
            gatherer, dAdp_full = PETSc.Scatter().toZero(dAdp)
            gatherer.scatter(dAdp, dAdp_full, False, PETSc.Scatter.Mode.FORWARD)

            # finish calculating the gradient
            if(NOT_PARALLEL):
                # derivative with respect to fields
                gradient[i] = np.sum(grad_full)

                # Next we compute the derivative with respect to eps and mu. We
                # exclude the PML regions because changes to the materials in
                # the PMLs are generally not something we want to consider.
                # TODO: make compatible with multiple update boxes...
                jmin = int(np.floor(ub[0]/X*N)); jmax = int(np.ceil(ub[1]/X*N))
                imin = int(np.floor(ub[2]/Y*M)); imax = int(np.ceil(ub[3]/Y*M))
                if(jmin < w_pml_l): jmin = w_pml_l
                if(jmax > N-w_pml_r): jmax = N-w_pml_r
                if(imin < w_pml_b): imin = w_pml_b
                if(imax > M-w_pml_t): imax = M-w_pml_t

                # note that the extraction of eps and mu from A must be handled
                # slightly differently in the TE and TM cases since the signs
                # along the diagonal are swapped and eps and mu are positioned
                # in different parts
                # NOTE: magic number 3 is number of field components
                if(isinstance(sim, fdfd.FDFD_TM)):
                    dmudp = dAdp_full[0::3].reshape([M,N]) * 1j
                    depsdp = dAdp_full[1::3].reshape([M,N]) / 1j
                elif(isinstance(sim, fdfd.FDFD_TE)):
                    depsdp = dAdp_full[0::3].reshape([M,N]) / 1j
                    dmudp = dAdp_full[1::3].reshape([M,N]) * 1j

                gradient[i] += np.real(
                               np.sum(dFdeps[imin:imax, jmin:jmax] * \
                                      depsdp[imin:imax, jmin:jmax]) + \
                               np.sum(dFdeps_conj[imin:imax, jmin:jmax] * \
                                      np.conj(depsdp[imin:imax, jmin:jmax])) + \
                               np.sum(dFdmu[imin:imax, jmin:jmax] * \
                                      dmudp[imin:imax, jmin:jmax]) + \
                               np.sum(dFdmu_conj[imin:imax, jmin:jmax] * \
                                      np.conj(dmudp[imin:imax, jmin:jmax])) \
                               )

            # revert the system to its original state
            params[i] = p0
            self.update_system(params)
            if(type(ub[0]) == list or type(ub[0]) == np.ndarray or \
               type(ub[0]) == tuple):
                for box in ub:
                    self.sim.update(box)
            else:
                self.sim.update(ub)

        if(NOT_PARALLEL):
            return gradient

class AdjointMethodPNF2D(AdjointMethodFM2D):
    """Define an AdjointMethod object for a figure of merit which contains
    power normalization in 2D problems.

    A power-normalized figure of merit has the form

    .. math::
        F(\\mathbf{E}, \\mathbf{H}, \\epsilon, \\mu) = \\frac{f(\\mathbf{E},
        \\mathbf{H})} {P_\mathrm{src}(\\mathbf{E}, \\mathbf{H}, \\epsilon, \\mu)}

    where :math:`\\epsilon` and :math:`\\mu` are the permittivity and
    permeability and :math:`f(...)` is a figure of merit which depends only on
    the fields (e.g. power flowing through a plane, mode match, etc)
    """

    def __init__(self, sim, step=1e-8):
        super(AdjointMethodPNF2D, self).__init__(sim, step)

    @abstractmethod
    def calc_f(self, sim, params):
        """Calculate the non-power-normalized figure of merit
        :math:`f(\\mathbf{E}, \\mathbf{H})`.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        float
            The value of the non-power-normalized figure of merit.
        """
        pass

    @abstractmethod
    def calc_dfdx(self, sim, params):
        """Calculate the derivative of the non-power-normalized figure of merit
        with respect to the fields in the discretized grid.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        tuple of numpy.ndarray
            The derivative of f with respect to the fields in the form (E, H)
        """
        pass

    def calc_penalty(self, sim, params):
        """Calculate the additive contribution to the figure of merit by
        explicit functions of the design variables.

        Because of the power normalization, we have to handle contributions to
        the figure of merit which depend explicitly on the design variables
        separately. This function returns the value of the functional Q(p)
        where Q(p) is given by F = f(E,H,p)/Psrc + Q(p).

        This is typically used to impose penalties to the figure of merit
        (hence the name of the function).

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        tuple of numpy.ndarray
            The value of the penalty function
        """
        return 0.0


    def calc_fom(self, sim, params):
        """Calculate the power-normalized figure of merit.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        float
            The value of the power-normalized figure of merit.
        """
        f = self.calc_f(sim, params)
        penalty = self.calc_penalty(sim, params)
        Psrc = sim.get_source_power()

        if(NOT_PARALLEL):
            return f / Psrc + penalty
        else:
            return None

    def calc_dFdx(self, sim, params):
        """Calculate the derivative of the power-normalized figure of merit
        with respect to the field.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        tuple of numpy.ndarray
            The derivative of F with respect to the fields in the form (E, H)
        """
        dfdx = self.calc_dfdx(sim, params)
        f = self.calc_f(sim, params)

        if(NOT_PARALLEL):
            if(isinstance(sim, fdfd.FDFD_TM)):
                dfdHz = dfdx[0]
                dfdEx = dfdx[1]
                dfdEy = dfdx[2]

                dFdHz, dFdEx, dFdEy = fomutils.power_norm_dFdx_TM(sim, f, dfdHz, \
                                                                          dfdEx, \
                                                                          dfdEy)
                return (dFdHz, dFdEx, dFdEy)
            elif(isinstance(sim, fdfd.FDFD_TE)):
                dfdEz = dfdx[0]
                dfdHx = dfdx[1]
                dfdHy = dfdx[2]

                dFdEz, dFdHx, dFdHy = fomutils.power_norm_dFdx_TE(sim, f, dfdEz, \
                                                                          dfdHx, \
                                                                          dfdHy)
                return (dFdEz, dFdHx, dFdHy)
        else:
            return None

    def calc_dFdm(self, sim, params):
        """Calculate the derivative of the power-normalized figure of merit
        with respect to the permittivity and permeability.

        Parameters
        ----------
        sim : emopt.fdfd.FDFD
            The FDFD simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        list of numpy.ndarray
            (Master node only) The derivative of F with respect to :math:`\epsilon`,
            :math:`\epsilon^*`, :math:`\mu`, and :math:`\mu^*`
        """
        # isinstance(sim, fdfd.FDFD_TM) must come before TE since TM is a
        # subclass of TE
        if(isinstance(sim, fdfd.FDFD_TM)):
            M = sim.M
            N = sim.N
            dx = sim.dx
            dy = sim.dy

            Hz = sim.get_field_interp('Hz')
            Ex = sim.get_field_interp('Ex')
            Ey = sim.get_field_interp('Ey')

            # compute the magnitudes squared of E and H -- this is all we need
            # here.
            if(NOT_PARALLEL):
                E2 = Ex * np.conj(Ex) + Ey * np.conj(Ey)
                H2 = Hz * np.conj(Hz)

        elif(isinstance(sim, fdfd.FDFD_TE)):
            M = sim.M
            N = sim.N
            dx = sim.dx
            dy = sim.dy

            Ez = sim.get_field_interp('Ez')
            Hx = sim.get_field_interp('Hx')
            Hy = sim.get_field_interp('Hy')

            # compute the magnitudes squared of E and H -- this is all we need
            # here.
            if(NOT_PARALLEL):
                E2 = Ez * np.conj(Ez)
                H2 = Hx * np.conj(Hx) + \
                     Hy * np.conj(Hy)

        if(NOT_PARALLEL):
            #y1 = eps, y2 = eps^*, y3 = mu, y4 = mu^*
            dPdy1 = -1j*0.125 * dx * dy * E2
            dPdy2 = 1j*0.125 * dx * dy * E2
            dPdy3 = -1j*0.125 * dx * dy * H2
            dPdy4 = 1j*0.125 * dx * dy * H2

            f = self.calc_f(sim, params)
            Ptot = sim.get_source_power()

            dFdy1 = -f / Ptot**2 * dPdy1
            dFdy2 = -f / Ptot**2 * dPdy2
            dFdy3 = -f / Ptot**2 * dPdy3
            dFdy4 = -f / Ptot**2 * dPdy4

            return dFdy1, dFdy2, dFdy3, dFdy4
        else:
            return None

class AdjointMethodPNF3D(AdjointMethod):
    """Define an AdjointMethod object for a figure of merit which contains
    power normalization in 3D problems.

    In 3D, lossy materials are not supported. As a result, power normalization
    is based purely on the power flux at the boundaries of the simulation (and
    is thus independent of the material values within the simulation domain).
    """

    def __init__(self, sim, step=1e-8):
        super(AdjointMethodPNF3D, self).__init__(sim, step)

    @abstractmethod
    def calc_f(self, sim, params):
        """Calculate the non-power-normalized figure of merit
        :math:`f(\\mathbf{E}, \\mathbf{H})`.

        Parameters
        ----------
        sim : emopt.simulation.MaxwellSolver
            The 3D simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        float
            The value of the non-power-normalized figure of merit.
        """
        pass

    @abstractmethod
    def calc_dfdx(self, sim, params):
        """Calculate the derivative of the non-power-normalized figure of merit
        with respect to the fields in the discretized grid.

        Parameters
        ----------
        sim : emopt.simulation.MaxwellSolver
            The 3D simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        list of tuples of 6 numpy.ndarray
            The derivative of f with respect to the fields in the form (E, H)
        """
        pass

    @abstractmethod
    def get_fom_domains(self):
        """Retrieve the user-defined domains where the figure of merit is
        calculated.

        Returns
        -------
        List of emopt.misc.DomainCoordinates
            The list of FOM domains.
        """
        pass

    def calc_penalty(self, sim, params):
        """Calculate the additive contribution to the figure of merit by
        explicit functions of the design variables.

        Because of the power normalization, we have to handle contributions to
        the figure of merit which depend explicitly on the design variables
        separately. This function returns the value of the functional Q(p)
        where Q(p) is given by F = f(E,H,p)/Psrc + Q(p).

        This is typically used to impose penalties to the figure of merit
        (hence the name of the function).

        Parameters
        ----------
        sim : emopt.simulation.MaxwellSolver
            The 3D simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        tuple of numpy.ndarray
            The value of the penalty function
        """
        return 0.0


    def calc_fom(self, sim, params):
        """Calculate the power-normalized figure of merit.

        Parameters
        ----------
        sim : emopt.simulation.MaxwellSolver
            The 3D simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        float
            The value of the power-normalized figure of merit.
        """
        f = self.calc_f(sim, params)
        penalty = self.calc_penalty(sim, params)
        Psrc = sim.source_power

        if(NOT_PARALLEL):
            return f / Psrc + penalty
        else:
            return None

    def calc_dFdx(self, sim, params):
        """Calculate the derivative of the power-normalized figure of merit
        with respect to the field.

        Parameters
        ----------
        sim : emopt.simulation.MaxwellSolver
            The 3D simulation object.
        params : numpy.ndarray
            The current set of design variables

        Returns
        -------
        tuple of numpy.ndarray
            The derivative of F with respect to the fields in the form (E, H)
        """
        fom = self.calc_f(sim, params)
        dfdxs = self.calc_dfdx(sim, params)
        domains = self.get_fom_domains()

        domains = COMM.bcast(domains, root=0)
        Nderiv = len(domains)

        adjoint_sources = [[], []]
        for i in range(Nderiv):
            if(NOT_PARALLEL):
                dfdx = dfdxs[i]
                dFdEx = dfdx[0]
                dFdEy = dfdx[1]
                dFdEz = dfdx[2]
                dFdHx = dfdx[3]
                dFdHy = dfdx[4]
                dFdHz = dfdx[5]
            else:
                dFdEx = None; dFdEy = None; dFdEz = None
                dFdHx = None; dFdHy = None; dFdHz = None

            fom_domain = domains[i]
            a_src = fomutils.power_norm_dFdx_3D(sim, fom,
                                                self.fom_domain,
                                                dFdEx, dFdEy, dFdEz,
                                                dFdHx, dFdHy, dFdHz)
            adjoint_sources = [adjoint_sources[0]+a_src[0],
                               adjoint_sources[1]+a_src[1]]
        return adjoint_sources
