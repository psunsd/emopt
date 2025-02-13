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

Common functions useful for calculating figures of merit and their derivatives.
"""
from __future__ import absolute_import
from builtins import range
from builtins import object
from . import fdfd, misc
from .misc import NOT_PARALLEL
from .defs import FieldComponent
import numpy as np

__author__ = "Andrew Michaels"
__license__ = "BSD-3"
__version__ = "2019.5.6"
__maintainer__ = "Andrew Michaels"
__status__ = "development"

#####################################################################################
# Step and Rect Functions
#####################################################################################

def step(x, k, y0=0, A=1.0):
    """Compute the value of a smooth and analytic step function.

    The step function is approximated using a logistic function which can be
    scaled and shifted:

    .. math::
        \\Pi(x) = \\frac{A}{1 + e^{-k x}} + y_0

    This function has the property that :math:`\\Pi \\rightarrow y_0` as :math:`x
    \\rightarrow -\infty` and :math:`\\Pi \\rightarrow y_0 + A` as :math:`x
    \\rightarrow \infty`.

    Parameters
    ----------
    x : float or numpy.ndarray
        The input values
    k : float
        The steepness of step function
    y0 : float (optional)
        The shift of the step function (default = 0)
    A : float (optional)
        The scale factor of the step function (default = 1.0)

    Returns
    -------
    float or numpy.ndarray
        The step function applied to x.
    """
    return A / (1 + np.exp(-k*x)) + y0

def step_derivative(x, k, y0=0, A=1.0):
    """Compute the derivative of a smooth approximation of a step function.

    Parameters
    ----------
    x : float or numpy.ndarray
        The input values
    k : float
        The steepness of step function
    y0 : float (optional)
        The shift of the step function (default = 0)
    A : float (optional)
        The scale factor of the step function (default = 1.0)

    Returns
    -------
    float or numpy.ndarray
        The dertivative of the step function applied to x.
    """
    exp_func = np.exp(-k*x)
    return A*k*exp_func / (1+exp_func)**2

def rect(x, w1p, ws):
    """Apply a smooth rect function.

    This function is centered at zero.

    Parameters
    ----------
    x : float or numpy.ndarray
        The values of the independent variables passed to the rect function.
    w1p : float
        The `one percent` width; i.e. the width w for which rect(+/- w/2) =
        0.01.
    ws : float
        The steepness of the sides of the rect function.

    Returns
    -------
    float or numpy.ndarray
        The value of rect(x).
    """
    k = 2*np.log(99.0)/ws
    x1 = -w1p/2.0 + ws/2
    x2 = w1p/2.0 - ws/2

    return 1/(1 + np.exp(-k*(x-x1))) - 1/(1 + np.exp(-k*(x-x2)))

def rect_derivative(x, w1p, ws):
    """Calculate the derivative of the smoothed rect function.

    Parameters
    ----------
    x : float or numpy.ndarray
        The values of the independent variables passed to the rect function.
    w1p : float
        The `one percent` width; i.e. the width w for which rect(+/- w/2) =
        0.01.
    ws : float
        The steepness of the sides of the rect function.

    Returns
    -------
    float or numpy.ndarray
        The value of :math:`d \mathrm{rect} / dt |_x`.
    """
    k = 2*np.log(99.0)/ws
    x1 = - w1p/2.0 + 1/k*np.log(99.0)
    x2 = + w1p/2.0 - 1/k*np.log(99.0)

    return k*np.exp(-k*(x-x1))/(1 + np.exp(-k*(x-x1)))**2 - k*np.exp(-k*(x-x2))/(1 + np.exp(-k*(x-x2)))**2

#####################################################################################
# Radius of Curvature Functions
#####################################################################################

def radius_of_curvature(x1, x2, x3, y1, y2, y3):
    """Compute the approximate radius of curvature of three points.

    This is achieved by first fitting a parabola to the three points and then
    finding the radius of cruvature of that parabola.

    Notes
    -----
    The radius of curvature is signed.

    Parameters
    ----------
    x1 : float
        The x coordinate of the first point
    x2 : float
        The x coordinate of the second point
    x3 : float
        The x coordinate of the third point
    y1 : float
        The y coordinate of the first point
    y2 : float
        The y coordinate of the second point
    y3 : float
        The y coordinate of the third point

    Returns
    -------
    float
        The approximate radius of curvature of the set of points
    """
    t0 = 0
    t1 = 0.5
    t2 = 1.0

    c = x1
    a = (x2 - (x3-x1)*t1 - x1)/(t1**2-t1)
    b = x3-x1-a

    f = y1
    d = (y2 - (y3-y1)*t1 - y1)/(t1**2-t1)
    e = y3-y1-d

    # We calculate the radius of curvature at point x2
    R = np.power((2*a*t1+b)**2 + (2*d*t1+e)**2, 1.5) / \
                ((2*a*t1+b)*2*d - (2*d*t1+e)*2*a)

    return float(R)

def d_roc_dx1(x1, x2, x3, y1, y2, y3):
    """Calculate the derivative of the radius of curvature with respect to the
    changes in the x coordinate of the point.

    This function calculates the derivative of the approximate radius of
    curvature at a point (x2, y2) with respect to x1.

    Notes
    -----
    Currently this is computed using a finite difference which will introduce a
    small amount of error and be slightly less performant than an analytic
    implementation. An analytic implementation will be implemented soon!

    Parameters
    ----------
    x1 : float
        The x coordinate of the first point
    x2 : float
        The x coordinate of the second point
    x3 : float
        The x coordinate of the third point
    y1 : float
        The y coordinate of the first point
    y2 : float
        The y coordinate of the second point
    y3 : float
        The y coordinate of the third point

    Returns
    -------
    float
        The derivative of the approximate radius of curvature at point (x2, y2)
        with respect to x1.
    """
    return 0.125*(12.0*(-x1 + x3)*((x1 - x3)*(2.0*y1 - 4.0*y2 + 2.0*y3) - \
           (y1 - y3)*(2.0*x1 - 4.0*x2 + 2.0*x3))*((x1 - x3)**2 + \
           (y1 - y3)**2)**0.5 - 16.0*(y2 - y3)*((x1 - x3)**2 + \
           (y1 - y3)**2)**1.5)/((x1 - x3)*(2.0*y1 - 4.0*y2 + 2.0*y3) - \
           (y1 - y3)*(2.0*x1 - 4.0*x2 + 2.0*x3))**2

def d_roc_dx2(x1, x2, x3, y1, y2, y3):
    """Calculate the derivative of the radius of curvature with respect to the
    changes in the x coordinate of the point.

    This function calculates the derivative of the approximate radius of
    curvature at a point (x2, y2) with respect to x2.

    Notes
    -----
    Currently this is computed using a finite difference which will introduce a
    small amount of error and be slightly less performant than an analytic
    implementation. An analytic implementation will be implemented soon!

    Parameters
    ----------
    x1 : float
        The x coordinate of the first point
    x2 : float
        The x coordinate of the second point
    x3 : float
        The x coordinate of the third point
    y1 : float
        The y coordinate of the first point
    y2 : float
        The y coordinate of the second point
    y3 : float
        The y coordinate of the third point

    Returns
    -------
    float
        The derivative of the approximate radius of curvature at point (x2, y2)
        with respect to x2.
    """
    return 2.0*(y1 - y3)*((x1 - x3)**2 + \
           (y1 - y3)**2)**1.5/((x1 - x3)*(2.0*y1 - 4.0*y2 + 2.0*y3) - \
           (y1 - y3)*(2.0*x1 - 4.0*x2 + 2.0*x3))**2

def d_roc_dx3(x1, x2, x3, y1, y2, y3):
    """Calculate the derivative of the radius of curvature with respect to the
    changes in the x coordinate of the point.

    This function calculates the derivative of the approximate radius of
    curvature at a point (x2, y2) with respect to x3.

    Notes
    -----
    Currently this is computed using a finite difference which will introduce a
    small amount of error and be slightly less performant than an analytic
    implementation. An analytic implementation will be implemented soon!

    Parameters
    ----------
    x1 : float
        The x coordinate of the first point
    x2 : float
        The x coordinate of the second point
    x3 : float
        The x coordinate of the third point
    y1 : float
        The y coordinate of the first point
    y2 : float
        The y coordinate of the second point
    y3 : float
        The y coordinate of the third point

    Returns
    -------
    float
        The derivative of the approximate radius of curvature at point (x2, y2)
        with respect to x3.
    """
    return 0.125*(12.0*(x1 - x3)*((x1 - x3)*(2.0*y1 - 4.0*y2 + 2.0*y3) - \
           (y1 - y3)*(2.0*x1 - 4.0*x2 + 2.0*x3))*((x1 - x3)**2 + \
           (y1 - y3)**2)**0.5 - 16.0*(y1 - y2)*((x1 - x3)**2 + \
           (y1 - y3)**2)**1.5)/((x1 - x3)*(2.0*y1 - 4.0*y2 + 2.0*y3) - \
           (y1 - y3)*(2.0*x1 - 4.0*x2 + 2.0*x3))**2

def d_roc_dy1(x1, x2, x3, y1, y2, y3):
    """Calculate the derivative of the radius of curvature with respect to the
    changes in the y coordinate of the point.

    This function calculates the derivative of the approximate radius of
    curvature at a point (x2, y2) with respect to y1.

    Notes
    -----
    Currently this is computed using a finite difference which will introduce a
    small amount of error and be slightly less performant than an analytic
    implementation. An analytic implementation will be implemented soon!

    Parameters
    ----------
    x1 : float
        The x coordinate of the first point
    x2 : float
        The x coordinate of the second point
    x3 : float
        The x coordinate of the third point
    y1 : float
        The y coordinate of the first point
    y2 : float
        The y coordinate of the second point
    y3 : float
        The y coordinate of the third point

    Returns
    -------
    float
        The derivative of the approximate radius of curvature at point (x2, y2)
        with respect to y1.
    """
    return 0.125*(16.0*(x2 - x3)*((x1 - x3)**2 + (y1 - y3)**2)**1.5 + \
           12.0*(-y1 + y3)*((x1 - x3)*(2.0*y1 - 4.0*y2 + 2.0*y3) - \
           (y1 - y3)*(2.0*x1 - 4.0*x2 + 2.0*x3))*((x1 - x3)**2 + \
           (y1 - y3)**2)**0.5)/((x1 - x3)*(2.0*y1 - 4.0*y2 + 2.0*y3) - \
           (y1 - y3)*(2.0*x1 - 4.0*x2 + 2.0*x3))**2

def d_roc_dy2(x1, x2, x3, y1, y2, y3):
    """Calculate the derivative of the radius of curvature with respect to the
    changes in the y coordinate of the point.

    This function calculates the derivative of the approximate radius of
    curvature at a point (x2, y2) with respect to y2.

    Notes
    -----
    Currently this is computed using a finite difference which will introduce a
    small amount of error and be slightly less performant than an analytic
    implementation. An analytic implementation will be implemented soon!

    Parameters
    ----------
    x1 : float
        The x coordinate of the first point
    x2 : float
        The x coordinate of the second point
    x3 : float
        The x coordinate of the third point
    y1 : float
        The y coordinate of the first point
    y2 : float
        The y coordinate of the second point
    y3 : float
        The y coordinate of the third point

    Returns
    -------
    float
        The derivative of the approximate radius of curvature at point (x2, y2)
        with respect to y2.
    """
    return 2.0*(-x1 + x3)*((x1 - x3)**2 + \
           (y1 - y3)**2)**1.5/((x1 - x3)*(2.0*y1 - 4.0*y2 + 2.0*y3) - \
           (y1 - y3)*(2.0*x1 - 4.0*x2 + 2.0*x3))**2

def d_roc_dy3(x1, x2, x3, y1, y2, y3):
    """Calculate the derivative of the radius of curvature with respect to the
    changes in the y coordinate of the point.

    This function calculates the derivative of the approximate radius of
    curvature at a point (x2, y2) with respect to y3.

    Notes
    -----
    Currently this is computed using a finite difference which will introduce a
    small amount of error and be slightly less performant than an analytic
    implementation. An analytic implementation will be implemented soon!

    Parameters
    ----------
    x1 : float
        The x coordinate of the first point
    x2 : float
        The x coordinate of the second point
    x3 : float
        The x coordinate of the third point
    y1 : float
        The y coordinate of the first point
    y2 : float
        The y coordinate of the second point
    y3 : float
        The y coordinate of the third point

    Returns
    -------
    float
        The derivative of the approximate radius of curvature at point (x2, y2)
        with respect to y3.
    """
    return 0.125*(16.0*(x1 - x2)*((x1 - x3)**2 + (y1 - y3)**2)**1.5 + \
           12.0*(y1 - y3)*((x1 - x3)*(2.0*y1 - 4.0*y2 + 2.0*y3) - \
           (y1 - y3)*(2.0*x1 - 4.0*x2 + 2.0*x3))*((x1 - x3)**2 + \
           (y1 - y3)**2)**0.5)/((x1 - x3)*(2.0*y1 - 4.0*y2 + 2.0*y3) - \
           (y1 - y3)*(2.0*x1 - 4.0*x2 + 2.0*x3))**2

def rocp(x, y, indices, Rmin, dR):
    """Calculate a penalty which acts as a minimum radius of curvature
    constraint.

    A radius of curvature constraint can be imposed by first calculating the
    approximate radius of curvature at every point and then penalizing a figure
    of merit when radii of curvature fall below a minimum value.  Penalization
    is achieved by applying a (smooth) rect function to the radii of curvature;
    when a radius is below a specified minimum, the resulting output of
    the function drops below zero, reducing the figure of merit.

    Parameters
    ----------
    x : numpy.ndarray
        The x coordinates of a polygon or connected set of points
    y : numpy.ndarray
        The y coordinates of a polygon or connected set of points
    indices : list or numpy.ndarray
        The list of array indices for which the radius of curvature is
        calculated
    Rmin : float
        The minimum radius of curvature
    dR : float
        The steepness of the step function used to determine violation of Rmin

    Returns
    -------
    float
        The values of the radius of curvature penalty function.
    """
    penalty = 0.0
    ps = []

    for i in indices:
        x1 = x[i-1]; x2 = x[i]; x3 = x[i+1]
        y1 = y[i-1]; y2 = y[i]; y3 = y[i+1]

        roc = radius_of_curvature(x1, x2, x3, y1, y2, y3)

        penalty += rect(roc, Rmin*2, dR)
        ps.append(rect(roc, Rmin*2, dR))

    return penalty

def rocp_derivative(x, y, indices, Rmin, dR):
    """Calculate the derivative of the radius of curvature penalty with respect
    to the set of (x,y) coordinates.

    Notes
    -----
    This function assumes that the indices are 'sorted', i.e., the indices
    correspond to moving clockwise or counter-clockwise around the line
    string/polygon. The function will not work if the points are out of order.

    Parameters
    ----------
    x : numpy.ndarray
        The x coordinates of a polygon or connected set of points
    y : numpy.ndarray
        The y coordinates of a polygon or connected set of points
    indices : list or numpy.ndarray
        The list of array indices for which the radius of curvature is
        calculated
    Rmin : float
        The minimum radius of curvature
    dR : float
        The steepness of the step function used to determine violation of Rmin

    Returns
    -------
    np.array, np.array
        Two lists containing d/dx(rocp) and d/dy(rocp)
    """
    Ni = len(indices)
    dPdx = np.zeros(Ni)
    dPdy = np.zeros(Ni)

    for i in range(Ni):
        j = indices[i]
        x1 = x[j-1]; x2 = x[j]; x3 = x[j+1]
        y1 = y[j-1]; y2 = y[j]; y3 = y[j+1]

        roc = radius_of_curvature(x1, x2, x3, y1, y2, y3)
        rect_deriv = rect_derivative(roc, 2*Rmin, dR)

        dPdx[i] += rect_deriv * d_roc_dx2(x1, x2, x3, y1, y2, y3)
        dPdy[i] += rect_deriv * d_roc_dy2(x1, x2, x3, y1, y2, y3)

        if(i > 0 and j-1 == indices[i-1]):
            dPdx[i-1] += rect_deriv * d_roc_dx1(x1, x2, x3, y1, y2, y3)
            dPdy[i-1] += rect_deriv * d_roc_dy1(x1, x2, x3, y1, y2, y3)

        if(i < Ni-1 and j+1 == indices[i+1]):
            dPdx[i+1] += rect_deriv * d_roc_dx3(x1, x2, x3, y1, y2, y3)
            dPdy[i+1] += rect_deriv * d_roc_dy3(x1, x2, x3, y1, y2, y3)

    return np.nan_to_num(dPdx), np.nan_to_num(dPdy)

#####################################################################################
# Bridge and Gap Size Functions
#####################################################################################

def ndisty(x, y, y0):
    """Calculate the approximate distance from each point in a polygon to a y
    position along the polygon normal direction.

    Given a polygon defined by a set of x and y coordinates, find the distance
    from each point in that polygon to a desired y position. The normal
    direction is approximated based on the surrounding points.s

    Parameters
    ---------
    x : list or np.array
        X coordinates of polygon
    y : list or np.array
        Y coordinates of polygon
    y0 : float
        The y position to which the distance is calculated.

    Returns
    -------
    np.array
        List of distances with length N-2 where N=len(x)
    """
    x = np.array(x)
    y = np.array(y)

    x1 = x[0:-2]; x2 = x[1:-1]; x3 = x[2:]
    y1 = y[0:-2]; y2 = y[1:-1]; y3 = y[2:]

    dx = x3-x1
    dy = y3-y1
    ds = np.sqrt(dx**2 + dy**2)
    nx = dy / ds
    ny = -dx / ds

    xb = x2 + nx/ny * (y0-y2)
    dists = np.sqrt((xb-x2)**2 + (y0-y2)**2)

    # account for parallel lines
    dists[ny == 0] = np.finfo('d').max

    return dists

def d_ndisty_dx1(x1, x2, x3, y1, y2, y3, y0):
    """Calculate the derivative of ndisty with respect the the x coordinate of
    the 'previous' point.

    Parameters
    ---------
    x1 : float or list or np.array
        X coordinate(s) of "previous" point
    x2 : float or list or np.array
        X coordinate(s) of "current" point
    x3 : float or list or np.array
        X coordinate(s) of "next" point
    y1 : float or list or np.array
        Y coordinate(s) of "previous" point
    y2 : float or list or np.array
        Y coordinate(s) of "current" point
    y3 : float or list or np.array
        Y coordinate(s) of "next" point
    y0 : float
        The y position to which the distance is calculated.

    Returns
    -------
    np.array
        List of derivatives with length N-2 where N=len(x)
    """
    return -(y0 - y2)**2*(y1 - y3)**2/(np.sqrt((y0 - y2)**2*((x1 - x3)**2 + \
            (y1 - y3)**2)/(x1 - x3)**2)*(x1 - x3)**3)

def d_ndisty_dx2(x1, x2, x3, y1, y2, y3, y0):
    """Calculate the derivative of ndisty with respect the the x coordinate of
    the 'current' point.

    Parameters
    ---------
    x1 : float or list or np.array
        X coordinate(s) of "previous" point
    x2 : float or list or np.array
        X coordinate(s) of "current" point
    x3 : float or list or np.array
        X coordinate(s) of "next" point
    y1 : float or list or np.array
        Y coordinate(s) of "previous" point
    y2 : float or list or np.array
        Y coordinate(s) of "current" point
    y3 : float or list or np.array
        Y coordinate(s) of "next" point
    y0 : float
        The y position to which the distance is calculated.

    Returns
    -------
    np.array
        List of derivatives with length N-2 where N=len(x)
    """
    return 0.0

def d_ndisty_dx3(x1, x2, x3, y1, y2, y3, y0):
    """Calculate the derivative of ndisty with respect the the x coordinate of
    the 'next' point.

    Parameters
    ---------
    x1 : float or list or np.array
        X coordinate(s) of "previous" point
    x2 : float or list or np.array
        X coordinate(s) of "current" point
    x3 : float or list or np.array
        X coordinate(s) of "next" point
    y1 : float or list or np.array
        Y coordinate(s) of "previous" point
    y2 : float or list or np.array
        Y coordinate(s) of "current" point
    y3 : float or list or np.array
        Y coordinate(s) of "next" point
    y0 : float
        The y position to which the distance is calculated.

    Returns
    -------
    np.array
        List of derivatives with length N-2 where N=len(x)
    """
    return (y0 - y2)**2*(y1 - y3)**2/(np.sqrt((y0 - y2)**2*((x1 - x3)**2 + \
           (y1 - y3)**2)/(x1 - x3)**2)*(x1 - x3)**3)

def d_ndisty_dy1(x1, x2, x3, y1, y2, y3, y0):
    """Calculate the derivative of ndisty with respect the the y coordinate of
    the 'previous' point.

    Parameters
    ---------
    x1 : float or list or np.array
        X coordinate(s) of "previous" point
    x2 : float or list or np.array
        X coordinate(s) of "current" point
    x3 : float or list or np.array
        X coordinate(s) of "next" point
    y1 : float or list or np.array
        Y coordinate(s) of "previous" point
    y2 : float or list or np.array
        Y coordinate(s) of "current" point
    y3 : float or list or np.array
        Y coordinate(s) of "next" point
    y0 : float
        The y position to which the distance is calculated.

    Returns
    -------
    np.array
        List of derivatives with length N-2 where N=len(x)
    """
    return (y0 - y2)**2*(y1 - y3)/(np.sqrt((y0 - y2)**2*((x1 - x3)**2 + \
           (y1 - y3)**2)/(x1 - x3)**2)*(x1 - x3)**2)

def d_ndisty_dy2(x1, x2, x3, y1, y2, y3, y0):
    """Calculate the derivative of ndisty with respect the the y coordinate of
    the 'current' point.

    Parameters
    ---------
    x1 : float or list or np.array
        X coordinate(s) of "previous" point
    x2 : float or list or np.array
        X coordinate(s) of "current" point
    x3 : float or list or np.array
        X coordinate(s) of "next" point
    y1 : float or list or np.array
        Y coordinate(s) of "previous" point
    y2 : float or list or np.array
        Y coordinate(s) of "current" point
    y3 : float or list or np.array
        Y coordinate(s) of "next" point
    y0 : float
        The y position to which the distance is calculated.

    Returns
    -------
    np.array
        List of derivatives with length N-2 where N=len(x)
    """
    return (-y0 + y2)*((x1 - x3)**2 + (y1 - y3)**2)/(np.sqrt((y0 - y2)**2 * \
           ((x1 - x3)**2 + (y1 - y3)**2)/(x1 - x3)**2)*(x1 - x3)**2)

def d_ndisty_dy3(x1, x2, x3, y1, y2, y3, y0):
    """Calculate the derivative of ndisty with respect the the y coordinate of
    the 'next' point.

    Parameters
    ---------
    x1 : float or list or np.array
        X coordinate(s) of "previous" point
    x2 : float or list or np.array
        X coordinate(s) of "current" point
    x3 : float or list or np.array
        X coordinate(s) of "next" point
    y1 : float or list or np.array
        Y coordinate(s) of "previous" point
    y2 : float or list or np.array
        Y coordinate(s) of "current" point
    y3 : float or list or np.array
        Y coordinate(s) of "next" point
    y0 : float
        The y position to which the distance is calculated.

    Returns
    -------
    np.array
        List of derivatives with length N-2 where N=len(x)
    """
    return (y0 - y2)**2*(-y1 + y3)/(np.sqrt((y0 - y2)**2*((x1 - x3)**2 + \
           (y1 - y3)**2)/(x1 - x3)**2)*(x1 - x3)**2)

def ndisty_penalty(x, y, y0, dmin, delta_d, inds=None):
    """Calculate a penalty function based on the minimum distance from each
    point to a y coordinate.

    This distance is calculated along the approximate normal direction of the
    polygon at each point. A thresholding function is applied to the distance
    in order to compare it to a minimum distance and hence determine a penalty
    value.

    This is useful as a simple approximate gap size constraint. For example, if
    a device is symmetric about y=0, you can call this function with y0=0 to
    penalize points which get too close to the y axis.

    Notes
    -----
    The inds parameter may not contain 0 or N-1 where N is the length of x and
    y.

    Parameters
    ---------
    x : list or np.array
        X coordinates of polygon
    y : list or np.array
        Y coordinates of polygon
    y0 : float
        The y position to which the distance is calculated.
    dmin : float
        The minimum distance.
    delta_d : float
        The steepness of the threshold function.
    inds : numpy.array (optional)
        The set of indices of x,y for which the penalty is calculated. If None,
        then the penalty will be calculated at all but the end points (default
        = None)

    Returns
    -------
    float
        The value of the penalty.
    """
    if(inds is None):
        inds = np.arange(1,len(x)-1)
    else:
        inds = np.array(inds)

    ndist = ndisty(x, y, y0)

    penalties = rect(ndist[inds-1], dmin*2, delta_d)
    return np.sum(penalties)

def ndisty_penalty_derivative(x, y, y0, dmin, delta_d, inds=None):
    """Calculate the derivative of ndisty_penalty with respect to the x and y
    coordinates.

    Notes
    -----
    The inds parameter may not contain 0 or N-1 where N is the length of x and
    y.

    Parameters
    ---------
    x : list or np.array
        X coordinates of polygon
    y : list or np.array
        Y coordinates of polygon
    y0 : float
        The y position to which the distance is calculated.
    dmin : float
        The minimum distance.
    delta_d : float
        The steepness of the threshold function.
    inds : numpy.array (optional)
        The set of indices of x,y for which the penalty is calculated. If None,
        then the penalty will be calculated at all but the end points (default
        = None)

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The derivatives with respect to the x and y coordinates, respectively.
    """
    if(inds is None):
        inds = np.arange(1,len(x)-1)
    else:
        inds = np.array(inds)

    Ni = len(inds)
    dPdx = np.zeros(Ni)
    dPdy = np.zeros(Ni)

    # Currently using a for loop in order to allow indices to be supplied in
    # any order. If points were supplied in increasing order, the we could
    # vectorize it :S
    for i in range(Ni):
        j = inds[i]
        x1 = x[j-1]; x2 = x[j]; x3 = x[j+1]
        y1 = y[j-1]; y2 = y[j]; y3 = y[j+1]

        ndist = ndisty([x1, x2, x3], [y1, y2, y3], y0)
        rect_deriv = rect_derivative(ndist, 2*dmin, delta_d)

        dPdx[i] += rect_deriv * d_ndisty_dx2(x1, x2, x3, y1, y2, y3, y0)
        dPdy[i] += rect_deriv * d_ndisty_dy2(x1, x2, x3, y1, y2, y3, y0)

        if(i > 0 and j-1 == inds[i-1]):
            dPdx[i-1] += rect_deriv * d_ndisty_dx1(x1, x2, x3, y1, y2, y3, y0)
            dPdy[i-1] += rect_deriv * d_ndisty_dy1(x1, x2, x3, y1, y2, y3, y0)

        if(i < Ni-1 and j+1 == inds[i+1]):
            dPdx[i+1] += rect_deriv * d_ndisty_dx3(x1, x2, x3, y1, y2, y3, y0)
            dPdy[i+1] += rect_deriv * d_ndisty_dy3(x1, x2, x3, y1, y2, y3, y0)

    return np.nan_to_num(dPdx), np.nan_to_num(dPdy)

def dist_to_edges(x1, x2, x3, y1, y2, y3, xe, ye):
    """Calculate the signed distance to a set of edges.

    Given a set of three points (x1, y1), (x2, y2), and (x3, y3) calculate the
    distance from (x2, y2) to the edges defined by lists of coordinates xe and
    ye. These distances typically correspond to gaps and bridges in a polygon.

    Parameters
    ----------
    x1 : float
        The x coordinate of the "previous" point
    y1 : float
        The y coordinate of the "previous" point
    x2 : float
        The x coordinate of the "current" point
    y2 : float
        The y coordinate of the "current" point
    x3 : float
        The x coordinate of the "next" point
    y3 : float
        The y coordinate of the "next" point
    xe : float
        The list of x coordinates which define a polygon
    ye: float
        The list of y coordinates which define a polygon

    Returns
    -------
    np.array
        The list of signed distances from point (x2, y2) to the edges defined
        by xe and ye. The number of distances will be equal to the number of
        edges in the polygon.
    np.array
        Parameters used to define edge lines. An intersection with an edge
        occurs when these parameters are between 0 and 1
    """
    ## Get the normal direction
    dx = x3-x1
    dy = y3-y1
    ds = np.sqrt(dx**2 + dy**2)
    nx = -dy / ds
    ny = dx / ds

    ## Loop through the edges and check for intersections
    dists = []
    us = []
    Ne = len(xe) - 1
    for j in range(Ne):
        xj0 = xe[j]; xj1 = xe[j+1]
        yj0 = ye[j]; yj1 = ye[j+1]

        dx = xj1 - xj0
        dy = yj1 - yj0

        ## check for parallel lines
        if(dy*nx - dx*ny == 0.0):
            return np.array([]), np.array([])

        t = (dx*(y2-yj0) - dy*(x2-xj0)) / (dy*nx - dx*ny)
        u = (nx*(yj0-y2) - ny*(xj0-x2)) / (ny*dx - nx*dy)

        xu = dx*u + xj0
        yu = dy*u + yj0

        dist = np.sqrt((xu-x2)**2 + (yu-y2)**2)

        ## sign the distance based on sign(t)
        # this can tell us if we have a bridge or gap
        dist *= np.sign(t)

        dists.append(dist)
        us.append(u)

    return np.array(dists), np.array(us)


#####################################################################################
# Mode Match
#####################################################################################


class ModeMatch(object):
    """Compute the mode match between two sets of electromagnetic fields.

    The mode match is essentially a projection of one set of fields onto a
    second set of fields. It defines the fraction of power in field 1 which
    propagates in field 2.

    When normalized with respect to the total source power injected into a system,
    this function can be used to compute coupling efficiencies.

    See [1] for a detailed derivation of the mode match equation.

    References
    ----------
    [1] A. Michaels, E. Yablonovitch, "Gradient-Based Inverse Electromagnetic Design
    Using Continuously-Smoothed Boundaries," Arxiv, 2017

    Parameters
    ----------
    normal : list or tuple
        The normal direction of the plane in which the mode match is computed
        in the form (x,y,z).
    ds1 : float
        The grid spacing along the first dimension of the supplied fields
    ds2 : float (optional)
        The grid spacing along the second dimension of the supplied fields. For
        1D mode matches, leave this untouched. (default = 1.0)
    Exm : numpy.ndarray (optional)
        The x component of the reference electric field
    Eym : numpy.ndarray (optional)
        The y component of the reference electric field
    Ezm : numpy.ndarray (optional)
        The z component of the reference electric field
    Hxm : numpy.ndarray (optional)
        The x component of the reference magnetic field
    Hym : numpy.ndarray (optional)
        The y component of the reference magnetic field
    Hzm : numpy.ndarray (optional)
        The z component of the reference magnetic field

    Methods
    -------
    compute(Ex=None, Ey=None, Ez=None, Hx=None, Hy=None, Hz=None)
        Compute the mode match and other underlying quantities.
    get_mode_match_forward(P_in)
        Get the mode match in the forward direction normalized with respect to
        a desired power.
    get_mode_match_back(P_in)
        Get the mode match in the backwards direction normalized with respect
        to a desired power.
    get_dFdEx()
        Get the derivative of unnormalized mode match with respect to Ex
    get_dFdEy()
        Get the derivative of unnormalized mode match with respect to Ey
    get_dFdEz()
        Get the derivative of unnormalized mode match with respect to Ez
    get_dFdHx()
        Get the derivative of unnormalized mode match with respect to Hx
    get_dFdHy()
        Get the derivative of unnormalized mode match with respect to Hy
    get_dFdHz()
        Get the derivative of unnormalized mode match with respect to Hz
    """

    def __init__(self, normal, ds1, ds2=1.0, Exm=None, Eym=None, Ezm=None,
                                             Hxm=None, Hym=None, Hzm=None):

        input_fields = [Exm, Eym, Ezm, Hxm, Hym, Hzm]

        self.fshape = None
        for f in input_fields:
            if(f is not None):
                self.fshape = f.shape[:]
                break

        if(self.fshape is None):
            raise ValueError('No fields were passed to ModeMatch.  Mode matching is impossible without fields!')

        self.Exm = Exm if Exm is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Eym = Eym if Eym is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Ezm = Ezm if Ezm is not None else np.zeros(self.fshape, dtype=np.complex128)

        self.Hxm = Hxm if Hxm is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Hym = Hym if Hym is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Hzm = Hzm if Hzm is not None else np.zeros(self.fshape, dtype=np.complex128)

        self.normal = np.array(normal)
        self.ds1 = ds1
        self.ds2 = ds2
        ds = ds1*ds2

        # cartesian basis vectors
        self.xhat = np.array([1, 0, 0])
        self.yhat = np.array([0, 1, 0])
        self.zhat = np.array([0, 0, 1])

        self.x_dot_s = self.xhat.dot(self.normal)
        self.y_dot_s = self.yhat.dot(self.normal)
        self.z_dot_s = self.zhat.dot(self.normal)

        # Calculate the mode field power normalization
        Pxm = self.Eym * np.conj(self.Hzm) - self.Ezm * np.conj(self.Hym)
        Pym = -self.Exm * np.conj(self.Hzm) + self.Ezm * np.conj(self.Hxm)
        Pzm = self.Exm * np.conj(self.Hym) - self.Eym * np.conj(self.Hxm)

        self.Pm = ds*np.sum(self.x_dot_s * Pxm + \
                            self.y_dot_s * Pym + \
                            self.z_dot_s * Pzm )

        self.efficiency = 0.0

        self.Ex = np.zeros(self.fshape, dtype=np.complex128)
        self.Ey = np.zeros(self.fshape, dtype=np.complex128)
        self.Ez = np.zeros(self.fshape, dtype=np.complex128)
        self.Hx = np.zeros(self.fshape, dtype=np.complex128)
        self.Hy = np.zeros(self.fshape, dtype=np.complex128)
        self.Hz = np.zeros(self.fshape, dtype=np.complex128)

    ## Some of the calculations are redundant, so we calculate most things in advance and 
    # save them for future access.
    def compute(self, Ex=None, Ey=None, Ez=None, Hx=None, Hy=None, Hz=None):
        """Compute the mode match and other underlying quantities.

        Notes
        -----
        This function must be called befor getting the mode match efficiency.

        If a NULL field is passed, it is assumed to be zero and have the
        correct dimensions.

        Parameters
        ----------
        Ex : numpy.ndarray (optional)
            The x component of the electric field. (default = None)
        Ey : numpy.ndarray (optional)
            The y component of the electric field. (default = None)
        Ez : numpy.ndarray (optional)
            The z component of the electric field. (default = None)
        Hx : numpy.ndarray (optional)
            The x component of the magnetic field. (default = None)
        Hy : numpy.ndarray (optional)
            The y component of the magnetic field. (default = None)
        Hz : numpy.ndarray (optional)
            The z component of the magnetic field. (default = None)
        """
        self.Ex = Ex if Ex is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Ey = Ey if Ey is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Ez = Ez if Ez is not None else np.zeros(self.fshape, dtype=np.complex128)

        self.Hx = Hx if Hx is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Hy = Hy if Hy is not None else np.zeros(self.fshape, dtype=np.complex128)
        self.Hz = Hz if Hz is not None else np.zeros(self.fshape, dtype=np.complex128)

        ds = self.ds1*self.ds2

        self.am1 = 0.5*ds*np.sum( (self.Ey*np.conj(self.Hzm) - self.Ez*np.conj(self.Hym))*self.x_dot_s - \
                                  (self.Ex*np.conj(self.Hzm) - self.Ez*np.conj(self.Hxm))*self.y_dot_s + \
                                  (self.Ex*np.conj(self.Hym) - self.Ey*np.conj(self.Hxm))*self.z_dot_s ) / self.Pm

        self.am2 = 0.5*ds*np.sum( (np.conj(self.Eym)*self.Hz - np.conj(self.Ezm)*self.Hy)*self.x_dot_s - \
                                  (np.conj(self.Exm)*self.Hz - np.conj(self.Ezm)*self.Hx)*self.y_dot_s + \
                                  (np.conj(self.Exm)*self.Hy - np.conj(self.Eym)*self.Hx)*self.z_dot_s ) / np.conj(self.Pm)

        self.am = self.am1 + self.am2
        self.bm = self.am1 - self.am2

        self.mode_match_fwd = 0.5 * self.am * np.conj(self.am) * np.real(self.Pm)
        self.mode_match_back = 0.5 * self.bm * np.conj(self.bm) * np.real(self.Pm)

    def get_mode_match_forward(self, P_in):
        """Get the mode match in the forward direction normalized with respect to
        a desired power.

        Parameters
        ----------
        P_in : float
            The power used to normalize the mode match.

        Returns
        -------
        float
            The mode match for forward-propagating fields.
        """
        return self.mode_match_fwd.real / P_in

    def get_mode_match_forward_phase(self):
        """Get the mode match phase in the forward direction.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The mode match phase for forward-propagating fields.
        """
        return np.angle(self.am)

    def get_mode_match_back(self, P_in):
        """Get the mode match in the backwards direction normalized with respect to
        a desired power.

        Parameters
        ----------
        P_in : float
            The power used to normalize the mode match.

        Returns
        -------
        float
            The mode match for backward-propagating fields.
        """
        return self.mode_match_back.real / P_in

    def get_mode_match_back_phase(self):
        """Get the mode match phase in the backward direction.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The mode match phase for backward-propagating fields.
        """
        return np.angle(self.bm)

    def get_dFdEx_phase(self):
        ds = self.ds1*self.ds2
        return 0.5*ds*(-np.conj(self.Hzm)*self.y_dot_s + np.conj(self.Hym)*self.z_dot_s) / self.Pm / 2.0 / 1.0j / self.am
    def get_dFdEy_phase(self):
        ds = self.ds1*self.ds2
        return 0.5*ds*(np.conj(self.Hzm)*self.x_dot_s - np.conj(self.Hxm)*self.z_dot_s) / self.Pm / 2.0 / 1.0j / self.am
    def get_dFdEz_phase(self):
        ds = self.ds1*self.ds2
        return 0.5*ds*(-np.conj(self.Hym)*self.x_dot_s + np.conj(self.Hxm)*self.y_dot_s) / self.Pm / 2.0 / 1.0j / self.am
    def get_dFdHx_phase(self):
        ds = self.ds1*self.ds2
        return 0.5*ds*(np.conj(self.Ezm)*self.y_dot_s - np.conj(self.Eym)*self.z_dot_s) / np.conj(self.Pm) / 2.0 / 1.0j / self.am
    def get_dFdHy_phase(self):
        ds = self.ds1*self.ds2
        return 0.5*ds*(-np.conj(self.Ezm)*self.x_dot_s + np.conj(self.Exm)*self.z_dot_s) / np.conj(self.Pm) / 2.0 / 1.0j / self.am
    def get_dFdHz_phase(self):
        ds = self.ds1*self.ds2
        return 0.5*ds*(np.conj(self.Eym)*self.x_dot_s - np.conj(self.Exm)*self.y_dot_s) / np.conj(self.Pm) / 2.0 / 1.0j / self.am

    def get_dFdEx(self):
        """Get the derivative of unnormalized mode match with respect to Ex.

        Returns
        -------
        numpy.ndarray
            The derivative of the unnormalized mode match with respect to the x
            component of the electric field.
        """
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (-np.conj(self.Hzm)*self.y_dot_s + np.conj(self.Hym)*self.z_dot_s) * np.conj(self.am)/self.Pm
    def get_dFdEy(self):
        """Get the derivative of unnormalized mode match with respect to Ey.

        Returns
        -------
        numpy.ndarray
            The derivative of the unnormalized mode match with respect to the y
            component of the electric field.
        """
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (np.conj(self.Hzm)*self.x_dot_s - np.conj(self.Hxm)*self.z_dot_s) * np.conj(self.am)/self.Pm
    def get_dFdEz(self):
        """Get the derivative of unnormalized mode match with respect to Ez.

        Returns
        -------
        numpy.ndarray
            The derivative of the unnormalized mode match with respect to the z
            component of the electric field.
        """
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (-np.conj(self.Hym)*self.x_dot_s + np.conj(self.Hxm)*self.y_dot_s) * np.conj(self.am)/self.Pm
    def get_dFdHx(self):
        """Get the derivative of unnormalized mode match with respect to Hx.

        Returns
        -------
        numpy.ndarray
            The derivative of the unnormalized mode match with respect to the x
            component of the magnetic field.
        """
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (np.conj(self.Ezm)*self.y_dot_s - np.conj(self.Eym)*self.z_dot_s) * np.conj(self.am)/np.conj(self.Pm)
    def get_dFdHy(self):
        """Get the derivative of unnormalized mode match with respect to Hy.

        Returns
        -------
        numpy.ndarray
            The derivative of the unnormalized mode match with respect to the y
            component of the magnetic field.
        """
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (-np.conj(self.Ezm)*self.x_dot_s + np.conj(self.Exm)*self.z_dot_s) * np.conj(self.am)/np.conj(self.Pm)
    def get_dFdHz(self):
        """Get the derivative of unnormalized mode match with respect to Hz.

        Returns
        -------
        numpy.ndarray
            The derivative of the unnormalized mode match with respect to the z
            component of the magnetic field.
        """
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (np.conj(self.Eym)*self.x_dot_s - np.conj(self.Exm)*self.y_dot_s) * np.conj(self.am)/np.conj(self.Pm)

    def get_dFdEx_back(self):
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (-np.conj(self.Hzm)*self.y_dot_s + np.conj(self.Hym)*self.z_dot_s) * np.conj(self.bm)/self.Pm
    def get_dFdEy_back(self):
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (np.conj(self.Hzm)*self.x_dot_s - np.conj(self.Hxm)*self.z_dot_s) * np.conj(self.bm)/self.Pm
    def get_dFdEz_back(self):
        ds = self.ds1*self.ds2
        return 1/4.0 * ds * np.real(self.Pm) * (-np.conj(self.Hym)*self.x_dot_s + np.conj(self.Hxm)*self.y_dot_s) * np.conj(self.bm)/self.Pm
    def get_dFdHx_back(self):
        ds = self.ds1*self.ds2
        return -1/4.0 * ds * np.real(self.Pm) * (np.conj(self.Ezm)*self.y_dot_s - np.conj(self.Eym)*self.z_dot_s) * np.conj(self.bm)/np.conj(self.Pm)
    def get_dFdHy_back(self):
        ds = self.ds1*self.ds2
        return -1/4.0 * ds * np.real(self.Pm) * (-np.conj(self.Ezm)*self.x_dot_s + np.conj(self.Exm)*self.z_dot_s) * np.conj(self.bm)/np.conj(self.Pm)
    def get_dFdHz_back(self):
        ds = self.ds1*self.ds2
        return -1/4.0 * ds * np.real(self.Pm) * (np.conj(self.Eym)*self.x_dot_s - np.conj(self.Exm)*self.y_dot_s) * np.conj(self.bm)/np.conj(self.Pm)

    def get_dFdEx_back_phase(self):
        ds = self.ds1*self.ds2
        return 0.5*ds*(-np.conj(self.Hzm)*self.y_dot_s + np.conj(self.Hym)*self.z_dot_s) / self.Pm / 2.0 / 1.0j / self.bm
    def get_dFdEy_back_phase(self):
        ds = self.ds1*self.ds2
        return 0.5*ds*(np.conj(self.Hzm)*self.x_dot_s - np.conj(self.Hxm)*self.z_dot_s) / self.Pm / 2.0 / 1.0j / self.bm
    def get_dFdEz_back_phase(self):
        ds = self.ds1*self.ds2
        return 0.5*ds*(-np.conj(self.Hym)*self.x_dot_s + np.conj(self.Hxm)*self.y_dot_s) / self.Pm / 2.0 / 1.0j / self.bm
    def get_dFdHx_back_phase(self):
        ds = self.ds1*self.ds2
        return -0.5*ds*(np.conj(self.Ezm)*self.y_dot_s - np.conj(self.Eym)*self.z_dot_s) / np.conj(self.Pm) / 2.0 / 1.0j / self.bm
    def get_dFdHy_back_phase(self):
        ds = self.ds1*self.ds2
        return -0.5*ds*(-np.conj(self.Ezm)*self.x_dot_s + np.conj(self.Exm)*self.z_dot_s) / np.conj(self.Pm) / 2.0 / 1.0j / self.bm
    def get_dFdHz_back_phase(self):
        ds = self.ds1*self.ds2
        return -0.5*ds*(np.conj(self.Eym)*self.x_dot_s - np.conj(self.Exm)*self.y_dot_s) / np.conj(self.Pm) / 2.0 / 1.0j / self.bm

#####################################################################################
# Functions of handling interpolated fields
#####################################################################################

def interpolated_dFdx_2D(sim, dFdEzi, dFdHxi, dFdHyi):
    """Account for interpolated fields in a 'naive' derivative of a figure of
    merit.

    In order to calculate any sort of quantity that involves power flow, we
    must interpolate the fields such that they are all known at the same point
    in space.  This process of interpolation must be handled very carefully in
    the contex of calculating gradients of a figure of merit.  In order to
    simplify this process and minimize the number of errors made, you can
    naively calculate the derivatives with respect to the interpolated fields
    and then compensate in order to ensure that the derivatives are correct
    with respect to the 'True' undelying shifted fields.

    Notes
    -----
    1. This function modifies the input arrays.

    2. dFdEz is not modified in the TE case

    Parameters
    ----------
    dFdEz : numpy.array
        2D numpy array containing derivative of FOM with respect to
        INTERPOLATED Ez
    dFdHx : numpy.array
        2D numpy array containing derivative of FOM with respect to
        INTERPOLATED Hx
    dFdHy : numpy.array
        2D numpy array containing derivative of FOM with respect to
        INTERPOLATED Hy

    Returns
    -------
    numpy.array, numpy.array, numpy.array
        The modified derivatives which account for interpolation
    """
    #dFdHx[0:-2, 1:-1] += dFdHx[1:-1, 1:-1]
    #dFdHy[1:-1, 2:] += dFdHy[1:-1, 1:-1]
    fshape = (dFdEzi.shape[0]+2, dFdEzi.shape[1]+2)

    dFdEz = np.zeros(fshape, dtype=np.complex128)
    dFdHx = np.zeros(fshape, dtype=np.complex128)
    dFdHy = np.zeros(fshape, dtype=np.complex128)

    dFdEz[1:-1, 1:-1] = dFdEzi
    dFdHx[1:-1, 1:-1] += dFdHxi/2.0
    dFdHx[0:-2, 1:-1] += dFdHxi/2.0
    dFdHy[1:-1, 1:-1] += dFdHyi/2.0
    dFdHy[1:-1, 2:] += dFdHyi/2.0

    dFdEz = dFdEz[1:-1, 1:-1]
    dFdHx = dFdHx[1:-1, 1:-1]
    dFdHy = dFdHy[1:-1, 1:-1]

    # Unfortunately, special boundary conditions complicate matters
    # It is easiest to handle this separately
    # Note only the first condition below has been tested..
    if(type(sim) == fdfd.FDFD_TM):
        if((sim.bc[1] == 'H' and type(sim) == fdfd.FDFD_TM) or \
           (sim.bc[1] == 'E' and type(sim) == fdfd.FDFD_TE)):
            dFdHx[0,:] = 0.0
        elif((sim.bc[1] == 'E' and type(sim) == fdfd.FDFD_TM) or \
             (sim.bc[1] == 'H' and type(sim) == fdfd.FDFD_TE)):
            dFdHx[0,:] = dFdHx[0,:]*2.0
        elif(sim.bc[1] == 'P'):
            dFdHx[0,:] += dFdHx[-1,:]/2.0
            dFdHx[-1,:] += dFdHx[0,:]/2.0

        if((sim.bc[0] == 'H' and type(sim) == fdfd.FDFD_TM) or \
           (sim.bc[0] == 'E' and type(sim) == fdfd.FDFD_TE)):
            dFdHy[:,-1] = 0.0
        elif((sim.bc[0] == 'E' and type(sim) == fdfd.FDFD_TM) or \
             (sim.bc[0] == 'H' and type(sim) == fdfd.FDFD_TE)):
            dFdHy[:,-1] = dFdHy[:,-1]*2.0
        elif(sim.bc[0] == 'P'):
            dFdHy[:,-1] += dFdHy[:,0]/2.0
            dFdHy[:,0] += dFdHy[:,-1]/2.0

    return dFdEz, dFdHx, dFdHy

def interpolated_dFdx_3D(sim, domain, dFdExi, dFdEyi, dFdEzi, dFdHxi, dFdHyi, dFdHzi):
    """Account for interpolated fields in a 'naive' derivative of a figure of
    merit.

    In order to calculate any sort of quantity that involves power flow, we
    must interpolate the fields such that they are all known at the same point
    in space.  This process of interpolation must be handled very carefully in
    the contex of calculating gradients of a figure of merit.  In order to
    simplify this process and minimize the number of errors made, you can
    naively calculate the derivatives with respect to the interpolated fields
    and then compensate in order to ensure that the derivatives are correct
    with respect to the 'True' undelying shifted fields.

    Notes
    -----
    Not all of the symmetry options have been tested. If you encounter issues
    with gradient accuracy, please post an issue on github!

    Parameters
    ----------
    sim : fdfd.FDFD_3D
        The simulation object--needed for boundary condition information
    domain : misc.DomainCoordinates
        The domain in which the derivative of the FOM has been computed
    dFdEx : numpy.array
        3D numpy array containing derivative of FOM with respect to
        INTERPOLATED Ex
    dFdEy : numpy.array
        3D numpy array containing derivative of FOM with respect to
        INTERPOLATED Ey
    dFdEz : numpy.array
        3D numpy array containing derivative of FOM with respect to
        INTERPOLATED Ez
    dFdHx : numpy.array
        2D numpy array containing derivative of FOM with respect to
        INTERPOLATED Hx
    dFdHy : numpy.array
        2D numpy array containing derivative of FOM with respect to
        INTERPOLATED Hy
    dFdHz : numpy.array
        2D numpy array containing derivative of FOM with respect to
        INTERPOLATED Hz

    Returns
    -------
    misc.DomainCoordinates, numpy.ndarray, numpy.ndarray, numpy.ndarray,
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        The modified derivatives which account for interpolation and the
        modified DomainCoordinates
    """
    # We will need to work with an array which is larger than the supplied
    # domain
    Nz, Ny, Nx = domain.shape

    Nx +=2; Ny += 2; Nz += 2
    shape = (Nz, Ny, Nx)

    # problem may be related to fact that there is averaging along x. This
    # means we need to add in a new x-normal plane...
    dFdEx = np.zeros(shape, dtype=np.complex128)
    dFdEx[1:-1, 1:-1, 1:-1] += dFdExi/4.0
    dFdEx[1:-1, 1:-1, 0:-2] += dFdExi/4.0
    dFdEx[2:, 1:-1, 1:-1]   += dFdExi/4.0
    dFdEx[2:, 1:-1, 0:-2]   += dFdExi/4.0

    # Handle boundary conditions for Ex
    if(domain.k1 == 0 and sim.bc[0] == 'H'):
        dFdEx[1:-1, 1:-1, 1] += dFdExi[:, :, 0]/4.0
        dFdEx[2:, 1:-1, 1] += dFdExi[:, :, 0]/4.0
    elif(domain.k1 == 0 and sim.bc[0] == 'E'):
        dFdEx[1:-1, 1, 1:-1] -= dFdExi[:, 0, :]/4.0
        dFdEx[2:, 1, 1:-1] -= dFdExi[:, 0, :]/4.0

    dFdEx = dFdEx[1:, 0:-1, 0:-1]

    dFdEy = np.zeros(shape, dtype=np.complex128)
    dFdEy[1:-1, 1:-1, 1:-1] += dFdEyi/4.0
    dFdEy[1:-1, 0:-2, 1:-1] += dFdEyi/4.0
    dFdEy[2:, 1:-1, 1:-1]   += dFdEyi/4.0
    dFdEy[2:, 0:-2, 1:-1]   += dFdEyi/4.0

    # Handle boundary conditions for Ey
    if(domain.j1 == 0 and sim.bc[1] == 'H'):
        dFdEy[1:-1, 1, 1:-1] += dFdEyi[:, 0, :]/4.0
        dFdEy[2:, 1, 1:-1] += dFdEyi[:, 0, :]/4.0
    elif(domain.j1 == 0 and sim.bc[1] == 'E'):
        dFdEy[1:-1, 1, 1:-1] -= dFdEyi[:, 0, :]/4.0
        dFdEy[2:, 1, 1:-1] -= dFdEyi[:, 0, :]/4.0

    dFdEy = dFdEy[1:, 0:-1, 0:-1]

    # Ez is Uninterpolated
    dFdEz = np.zeros(shape, dtype=np.complex128)
    dFdEz[1:-1, 1:-1, 1:-1] = dFdEzi
    dFdEz = dFdEz[1:, 0:-1, 0:-1]

    # This one might have a problem... It leads to a higher gradient error in
    # initial tests 
    dFdHx = np.zeros(shape, dtype=np.complex128)
    dFdHx[1:-1, 1:-1, 1:-1] += dFdHxi/2.0
    dFdHx[1:-1, 0:-2, 1:-1] += dFdHxi/2.0

    # Handle boundary conditions for Hx
    if(domain.j1 == 0 and sim.bc[1] == 'H'):
        dFdHx[1:-1, 1, 1:-1] += dFdHxi[:, 0, :]/2.0
    elif(domain.j1 == 0 and sim.bc[1] == 'E'):
        dFdHx[1:-1, 1, 1:-1] -= dFdHxi[:, 0, :]/2.0

    dFdHx = dFdHx[1:, 0:-1, 0:-1]

    dFdHy = np.zeros(shape, dtype=np.complex128)
    dFdHy[1:-1, 1:-1, 1:-1] += dFdHyi/2.0
    dFdHy[1:-1, 1:-1, 0:-2] += dFdHyi/2.0
    dFdHy = dFdHy[1:, 0:-1, 0:-1]

    # This one might have a problem... It leads to a higher gradient error in
    # initial tests
    dFdHz = np.zeros(shape, dtype=np.complex128)
    dFdHz[1:-1, 1:-1, 1:-1] += dFdHzi/8.0
    dFdHz[1:-1, 0:-2, 1:-1] += dFdHzi/8.0
    dFdHz[1:-1, 1:-1, 0:-2] += dFdHzi/8.0
    dFdHz[1:-1, 0:-2, 0:-2] += dFdHzi/8.0
    dFdHz[2:, 1:-1, 1:-1]   += dFdHzi/8.0
    dFdHz[2:, 0:-2, 1:-1]   += dFdHzi/8.0
    dFdHz[2:, 1:-1, 0:-2]   += dFdHzi/8.0
    dFdHz[2:, 0:-2, 0:-2]   += dFdHzi/8.0

    # Handle boundary conditions for Hz
    if(domain.j1 == 0 and sim.bc[1] == 'H'):
        dFdHz[1:-1, 1, 1:-1] += dFdHzi[:, 0, :]/8.0
        dFdHz[1:-1, 1, 0:-2] += dFdHzi[:, 0, :]/8.0
        dFdHz[2:, 1, 1:-1]   += dFdHzi[:, 0, :]/8.0
        dFdHz[2:, 1, 0:-2]   += dFdHzi[:, 0, :]/8.0
    elif(domain.j1 == 0 and sim.bc[1] == 'E'):
        dFdHz[1:-1, 1, 1:-1] -= dFdHzi[:, 0, :]/8.0
        dFdHz[1:-1, 1, 0:-2] -= dFdHzi[:, 0, :]/8.0
        dFdHz[2:, 1, 1:-1]   -= dFdHzi[:, 0, :]/8.0
        dFdHz[2:, 1, 0:-2]   -= dFdHzi[:, 0, :]/8.0

    dFdHz = dFdHz[1:, 0:-1, 0:-1]

    domain = domain.copy()
    domain.grow(1,0,1,0,0,1)

    return domain, dFdEx, dFdEy, dFdEz, dFdHx, dFdHy, dFdHz

#####################################################################################
# Functions of handling power normalization
#####################################################################################

def power_norm_dFdx_TE(sim, f, dfdEz, dfdHx, dfdHy):
    """Compute the derivative of a figure of merit which has power
    normalization.

    In many if not most cases of electromagnetic optimization, we will consider
    optimization problems in which we want to maximize some quantity which is
    normalized with respect to the total source power leaving a source,
    i.e.

            F(E,H) = f(E,H)/Psrc

    In some cases (e.g. a dipole emitting into a dielectric structure), the
    source power will depend on the shape of the structure itself. As a result,
    we need to account for this in our gradient calculations.

    Fortunately, we can easily account for this by writing our merit function
    as

            F(E,H) = f(E,H)/Psrc(E,H)

    and taking the necessary derivatives of Psrc. This process requires no deep
    knowledge about f.  It only needs f and its (numerical) derivative.

    Notes
    -----
    1. This function assumes f(E,H) is a function of the INTERPOLATED fields.
    All interpolation compensation is taken care of here (so don't call
    interp_dFdx on the dfdx's!!)

    Parameters
    ----------
    sim : emopt.fdfd.FDFD_TE
        simulation object which is needed in order to access field components
        as well as grid parameters
    f : float
        current value of merit function
    dfdEz : numpy.ndarray
        derivative w.r.t. Ez of non-normalized figure of merit
    dfdHx : numpy.ndarray
        derivative w.r.t. Hx of non-normalized figure of merit
    dfdHy : numpy.ndarray
        derivative w.r.t. Hy of non-normalized figure of merit

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        The derivative of the full power-normalized figure of merit with
        interpolation accounted for.
    """
    Ezc = np.conj(sim.get_field_interp('Ez'))
    Hxc = np.conj(sim.get_field_interp('Hx'))
    Hyc = np.conj(sim.get_field_interp('Hy'))
    Psrc = sim.get_source_power()
    dx = sim.dx
    dy = sim.dy
    M = sim.M
    N = sim.N

    if(not sim.real_materials):
        eps = sim.eps.get_values(0,N,0,M)
        mu = sim.mu.get_values(0,N,0,M)
    else:
        eps = np.zeros(Ezc.shape, dtype=np.complex128)
        mu = np.zeros(Ezc.shape, dtype=np.complex128)

    # get the planes through which power leaves the system
    w_pml_l = sim._w_pml_left
    w_pml_r = sim._w_pml_right
    w_pml_t = sim._w_pml_top
    w_pml_b = sim._w_pml_bottom

    x_bot = np.arange(w_pml_l, N-w_pml_r)
    y_bot = w_pml_b
    x_top = np.arange(w_pml_l, N-w_pml_r)
    y_top = M-w_pml_t

    x_left = w_pml_l
    y_left = np.arange(w_pml_b, M-w_pml_t)
    x_right = N-w_pml_r
    y_right = np.arange(w_pml_b, M-w_pml_t)

    x_all = np.arange(w_pml_l, N-w_pml_r)
    y_all = np.arange(w_pml_b, M-w_pml_t)
    y_all = y_all.reshape(y_all.shape[0], 1).astype(np.int)

    dPdEz = np.zeros([M, N], dtype=np.complex128)
    dPdHx = np.zeros([M, N], dtype=np.complex128)
    dPdHy = np.zeros([M, N], dtype=np.complex128)

    dPdEz[y_left, x_left]   += 0.25*dy*Hyc[y_left, x_left]
    dPdEz[y_top, x_top]     += 0.25*dx*Hxc[y_top, x_top]
    dPdEz[y_right, x_right] += -0.25*dy*Hyc[y_right, x_right]
    dPdEz[y_bot, x_bot]     += -0.25*dx*Hxc[y_bot, x_bot]
    dPdEz[y_all, x_all]     += 0.25*dx*dy*eps[y_all,x_all].imag*Ezc[y_all, x_all]

    dPdHx[y_top, x_top] += 0.25*dx*Ezc[y_top, x_top]
    dPdHx[y_bot, x_bot] += -0.25*dx*Ezc[y_bot, x_bot]
    dPdHx[y_all, x_all] += 0.25*dx*dy*mu[y_all,x_all].imag*Hxc[y_all, x_all]

    dPdHy[y_left, x_left]   += 0.25*dy*Ezc[y_left, x_left]
    dPdHy[y_right, x_right] += -0.25*dy*Ezc[y_right, x_right]
    dPdHy[y_all, x_all] += 0.25*dx*dy*mu[y_all,x_all].imag*Hyc[y_all, x_all]

    dFdEz = (Psrc * dfdEz - f * dPdEz) / Psrc**2
    dFdHx = (Psrc * dfdHx - f * dPdHx) / Psrc**2
    dFdHy = (Psrc * dfdHy - f * dPdHy) / Psrc**2

    dFdEz, dFdHx, dFdHy = interpolated_dFdx_2D(sim, dFdEz, dFdHx, dFdHy)

    return dFdEz, dFdHx, dFdHy

def power_norm_dFdx_TM(sim, f, dfdHz, dfdEx, dfdEy):
    """Compute the derivative of a figure of merit which has power
    normalization.

    In many if not most cases of electromagnetic optimization, we will consider
    optimization problems in which we want to maximize some quantity which is
    normalized with respect to the total source power leaving a source,
    i.e.

            F(E,H) = f(E,H)/Psrc

    In some cases (e.g. a dipole emitting into a dielectric structure), the
    source power will depend on the shape of the structure itself. As a result,
    we need to account for this in our gradient calculations.

    Fortunately, we can easily account for this by writing our merit function
    as

            F(E,H) = f(E,H)/Psrc(E,H)

    and taking the necessary derivatives of Psrc. This process requires no deep
    knowledge about f.  It only needs f and its (numerical) derivative.

    Notes
    -----
    1. This function assumes f(E,H) is a function of the INTERPOLATED fields.
    All interpolation compensation is taken care of here (so don't call
    interp_dFdx on the dfdx's!!)

    Parameters
    ----------
    sim : emopt.fdfd.FDFD_TM
        simulation object which is needed in order to access field components
        as well as grid parameters
    f : float
        current value of merit function
    dfdEz : numpy.ndarray
        derivative w.r.t. Ez of non-normalized figure of merit
    dfdHx : numpy.ndarray
        derivative w.r.t. Hx of non-normalized figure of merit
    dfdHy : numpy.ndarray
        derivative w.r.t. Hy of non-normalized figure of merit

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        The derivative of the full power-normalized figure of merit with
        interpolation accounted for.
    """
    Hzc = np.conj(sim.get_field_interp('Hz'))
    Exc = np.conj(sim.get_field_interp('Ex'))
    Eyc = np.conj(sim.get_field_interp('Ey'))
    Psrc = sim.get_source_power()
    dx = sim.dx
    dy = sim.dy
    M = sim.M
    N = sim.N

    if(not sim.real_materials):
        eps = sim.eps.get_values(0,N,0,M)
        mu = sim.mu.get_values(0,N,0,M)
    else:
        eps = np.zeros(Hzc.shape, dtype=np.complex128)
        mu = np.zeros(Hzc.shape, dtype=np.complex128)

    # get the planes through which power leaves the system
    w_pml_l = sim._w_pml_left
    w_pml_r = sim._w_pml_right
    w_pml_t = sim._w_pml_top
    w_pml_b = sim._w_pml_bottom

    x_bot = np.arange(w_pml_l, N-w_pml_r)
    y_bot = w_pml_b
    x_top = np.arange(w_pml_l, N-w_pml_r)
    y_top = M-w_pml_t

    x_left = w_pml_l
    y_left = np.arange(w_pml_b, M-w_pml_t)
    x_right = N-w_pml_r
    y_right = np.arange(w_pml_b, M-w_pml_t)

    x_all = np.arange(w_pml_l, N-w_pml_r)
    y_all = np.arange(w_pml_b, M-w_pml_t)
    y_all = y_all.reshape(y_all.shape[0], 1).astype(np.int)

    dPdHz = np.zeros([M, N], dtype=np.complex128)
    dPdEx = np.zeros([M, N], dtype=np.complex128)
    dPdEy = np.zeros([M, N], dtype=np.complex128)

    dPdHz[y_left, x_left]   += -0.25*dy*Eyc[y_left, x_left]
    dPdHz[y_top, x_top]     += -0.25*dx*Exc[y_top, x_top]
    dPdHz[y_right, x_right] += 0.25*dy*Eyc[y_right, x_right]
    dPdHz[y_bot, x_bot]     += 0.25*dx*Exc[y_bot, x_bot]
    dPdHz[y_all, x_all]     += 0.25*dx*dy*mu[y_all,x_all].imag*Hzc[y_all, x_all]

    dPdEx[y_top, x_top] += -0.25*dx*Hzc[y_top, x_top]
    dPdEx[y_bot, x_bot] += +0.25*dx*Hzc[y_bot, x_bot]
    dPdEx[y_all, x_all] += 0.25*dx*dy*eps[y_all,x_all].imag*Exc[y_all, x_all]

    dPdEy[y_left, x_left]   += -0.25*dy*Hzc[y_left, x_left]
    dPdEy[y_right, x_right] += 0.25*dy*Hzc[y_right, x_right]
    dPdEy[y_all, x_all] += 0.25*dx*dy*eps[y_all,x_all].imag*Eyc[y_all, x_all]

    dFdHz = (Psrc * dfdHz - f * dPdHz) / Psrc**2
    dFdEx = (Psrc * dfdEx - f * dPdEx) / Psrc**2
    dFdEy = (Psrc * dfdEy - f * dPdEy) / Psrc**2

    dFdHz, dFdEx, dFdEy = interpolated_dFdx_2D(sim, dFdHz, dFdEx, dFdEy)

    return dFdHz, dFdEx, dFdEy

def power_norm_dFdx_3D(sim, f, domain, dfdEx, dfdEy, dfdEz, dfdHx, dfdHy, dfdHz):
    """Compute the derivative of a figure of merit which has power
    normalization.

    The behavior of this function is very similar to its 2D counterparts.

    Notes
    -----
    Currently, this function does NOT account for the fact that the fields are
    interpolated. This will lead to some finite amount of error in the gradient
    calculations. In most cases, this increased error should not cause any
    issues.

    Parameters
    ----------
    sim : emopt.fdfd.FDFD_3D
        simulation object which is needed in order to access field components
        as well as grid parameters
    f : float
        current value of merit function
    dfdEx : numpy.ndarray
        derivative w.r.t. Ex of non-normalized figure of merit
    dfdEy : numpy.ndarray
        derivative w.r.t. Ey of non-normalized figure of merit
    dfdEz : numpy.ndarray
        derivative w.r.t. Ez of non-normalized figure of merit
    dfdHx : numpy.ndarray
        derivative w.r.t. Hx of non-normalized figure of merit
    dfdHy : numpy.ndarray
        derivative w.r.t. Hy of non-normalized figure of merit
    dfdHz : numpy.ndarray
        derivative w.r.t. Hy of non-normalized figure of merit

    Returns
    -------
    list
        List of source arrays and domains in the format needed by
        emotp.fdfd.FDFD_3D.set_adjoint_source(src)
    """
    Psrc = sim.source_power

    adj_sources = []
    adj_domains = []

    if(NOT_PARALLEL):
        dfdEH = interpolated_dFdx_3D(sim, domain,
                                     dfdEx/Psrc, dfdEy/Psrc, dfdEz/Psrc,
                                     dfdHx/Psrc, dfdHy/Psrc, dfdHz/Psrc)
        adj_sources.append(dfdEH[1:]); adj_domains.append(dfdEH[0])

    # setup the domains that are needed to get the power flux
    w_pml = sim.w_pml
    w_pml_xmin = w_pml[0]
    w_pml_xmax = w_pml[1]
    w_pml_ymin = w_pml[2]
    w_pml_ymax = w_pml[3]
    w_pml_zmin = w_pml[4]
    w_pml_zmax = w_pml[5]
    dx = sim.dx; dy = sim.dy; dz = sim.dz
    X = sim.X; Y = sim.Y; Z = sim.Z

    if(w_pml_xmin > 0): xmin = w_pml_xmin + dx
    else: xmin = 0.0

    if(w_pml_xmax > 0): xmax = X - w_pml_xmax - dx
    else: xmax = X

    if(w_pml_ymin > 0): ymin = w_pml_ymin + dy
    else: ymin = 0.0

    if(w_pml_ymax > 0): ymax = Y - w_pml_ymax - dy
    else: ymax = Y

    if(w_pml_zmin > 0): zmin = w_pml_zmin + dz
    else: zmin = 0.0

    if(w_pml_zmax > 0): zmax = Z - w_pml_zmax - dz
    else: zmax = Z

    x1 = misc.DomainCoordinates(xmin, xmin, ymin, ymax, zmin, zmax, dx, dy, dz)
    x2 = misc.DomainCoordinates(xmax, xmax, ymin, ymax, zmin, zmax, dx, dy, dz)
    y1 = misc.DomainCoordinates(xmin, xmax, ymin, ymin, zmin, zmax, dx, dy, dz)
    y2 = misc.DomainCoordinates(xmin, xmax, ymax, ymax, zmin, zmax, dx, dy, dz)
    z1 = misc.DomainCoordinates(xmin, xmax, ymin, ymax, zmin, zmin, dx, dy, dz)
    z2 = misc.DomainCoordinates(xmin, xmax, ymin, ymax, zmax, zmax, dx, dy, dz)

    # calculate dFdx for xmin
    dshape = x1.shape
    Ey = sim.get_field_interp(FieldComponent.Ey, domain=x1)
    Ez = sim.get_field_interp(FieldComponent.Ez, domain=x1)
    Hy = sim.get_field_interp(FieldComponent.Hy, domain=x1)
    Hz = sim.get_field_interp(FieldComponent.Hz, domain=x1)

    if(NOT_PARALLEL and sim.bc[0] != 'E' and sim.bc[0] != 'H'):
        dSdEx = np.zeros(dshape, dtype=np.complex128)
        dSdEy = -0.25 * dy * dz * np.conj(Hz)
        dSdEz = 0.25 * dy * dz * np.conj(Hy)
        dSdHx = np.zeros(dshape, dtype=np.complex128)
        dSdHy = 0.25 * dy * dz * np.conj(Ez)
        dSdHz = -0.25 * dy * dz * np.conj(Ey)

        dPdEx = -1 * f * dSdEx / Psrc**2
        dPdEy = -1 * f * dSdEy / Psrc**2
        dPdEz = -1 * f * dSdEz / Psrc**2
        dPdHx = -1 * f * dSdHx / Psrc**2
        dPdHy = -1 * f * dSdHy / Psrc**2
        dPdHz = -1 * f * dSdHz / Psrc**2

        dPdEH = interpolated_dFdx_3D(sim, x1, dPdEx, dPdEy, dPdEz,
                                     dPdHx, dPdHy, dPdHz)

        adj_sources.append(dPdEH[1:]); adj_domains.append(dPdEH[0])

    # clean up
    del Ey; del Ez; del Hy; del Hz

    # calculate dFdx for xmax
    dshape = x2.shape
    Ey = sim.get_field_interp(FieldComponent.Ey, domain=x2)
    Ez = sim.get_field_interp(FieldComponent.Ez, domain=x2)
    Hy = sim.get_field_interp(FieldComponent.Hy, domain=x2)
    Hz = sim.get_field_interp(FieldComponent.Hz, domain=x2)

    if(NOT_PARALLEL):
        dSdEx = np.zeros(dshape, dtype=np.complex128)
        dSdEy = 0.25 * dy * dz * np.conj(Hz)
        dSdEz = -0.25 * dy * dz * np.conj(Hy)
        dSdHx = np.zeros(dshape, dtype=np.complex128)
        dSdHy = -0.25 * dy * dz * np.conj(Ez)
        dSdHz = 0.25 * dy * dz * np.conj(Ey)

        dPdEx = -1 * f * dSdEx / Psrc**2
        dPdEy = -1 * f * dSdEy / Psrc**2
        dPdEz = -1 * f * dSdEz / Psrc**2
        dPdHx = -1 * f * dSdHx / Psrc**2
        dPdHy = -1 * f * dSdHy / Psrc**2
        dPdHz = -1 * f * dSdHz / Psrc**2

        dPdEH = interpolated_dFdx_3D(sim, x2, dPdEx, dPdEy, dPdEz,
                                     dPdHx, dPdHy, dPdHz)

        adj_sources.append(dPdEH[1:]); adj_domains.append(dPdEH[0])

    # clean up
    del Ey; del Ez; del Hy; del Hz

    # calculate dFdx for ymin
    dshape = y1.shape
    Ex = sim.get_field_interp(FieldComponent.Ex, domain=y1)
    Ez = sim.get_field_interp(FieldComponent.Ez, domain=y1)
    Hx = sim.get_field_interp(FieldComponent.Hx, domain=y1)
    Hz = sim.get_field_interp(FieldComponent.Hz, domain=y1)

    if(NOT_PARALLEL and sim.bc[1] != 'E' and sim.bc[1] != 'H'):
        dSdEx = 0.25 * dx * dz * np.conj(Hz)
        dSdEy = np.zeros(dshape, dtype=np.complex128)
        dSdEz = -0.25 * dx * dz * np.conj(Hx)
        dSdHx = -0.25 * dx * dz * np.conj(Ez)
        dSdHy = np.zeros(dshape, dtype=np.complex128)
        dSdHz = 0.25 * dx * dz * np.conj(Ex)

        dPdEx = -1 * f * dSdEx / Psrc**2
        dPdEy = -1 * f * dSdEy / Psrc**2
        dPdEz = -1 * f * dSdEz / Psrc**2
        dPdHx = -1 * f * dSdHx / Psrc**2
        dPdHy = -1 * f * dSdHy / Psrc**2
        dPdHz = -1 * f * dSdHz / Psrc**2

        dPdEH = interpolated_dFdx_3D(sim, y1, dPdEx, dPdEy, dPdEz,
                                     dPdHx, dPdHy, dPdHz)

        adj_sources.append(dPdEH[1:]); adj_domains.append(dPdEH[0])

    # clean up
    del Ex; del Ez; del Hx; del Hz

    # calculate dFdx for ymax
    dshape = y2.shape
    Ex = sim.get_field_interp(FieldComponent.Ex, domain=y2)
    Ez = sim.get_field_interp(FieldComponent.Ez, domain=y2)
    Hx = sim.get_field_interp(FieldComponent.Hx, domain=y2)
    Hz = sim.get_field_interp(FieldComponent.Hz, domain=y2)

    if(NOT_PARALLEL):
        dSdEx = -0.25 * dx * dz * np.conj(Hz)
        dSdEy = np.zeros(dshape, dtype=np.complex128)
        dSdEz = 0.25 * dx * dz * np.conj(Hx)
        dSdHx = 0.25 * dx * dz * np.conj(Ez)
        dSdHy = np.zeros(dshape, dtype=np.complex128)
        dSdHz = -0.25 * dx * dz * np.conj(Ex)

        dPdEx = -1 * f * dSdEx / Psrc**2
        dPdEy = -1 * f * dSdEy / Psrc**2
        dPdEz = -1 * f * dSdEz / Psrc**2
        dPdHx = -1 * f * dSdHx / Psrc**2
        dPdHy = -1 * f * dSdHy / Psrc**2
        dPdHz = -1 * f * dSdHz / Psrc**2

        dPdEH = interpolated_dFdx_3D(sim, y2, dPdEx, dPdEy, dPdEz,
                                     dPdHx, dPdHy, dPdHz)

        adj_sources.append(dPdEH[1:]); adj_domains.append(dPdEH[0])

    # clean up
    del Ex; del Ez; del Hx; del Hz

    # calculate dFdx for zmin
    dshape = z1.shape
    Ex = sim.get_field_interp(FieldComponent.Ex, domain=z1)
    Ey = sim.get_field_interp(FieldComponent.Ey, domain=z1)
    Hx = sim.get_field_interp(FieldComponent.Hx, domain=z1)
    Hy = sim.get_field_interp(FieldComponent.Hy, domain=z1)

    if(NOT_PARALLEL and sim.bc[2] != 'E' and sim.bc[2] != 'H'):
        dSdEx = -0.25 * dx * dy * np.conj(Hy)
        dSdEy = 0.25 * dx * dy * np.conj(Hx)
        dSdEz = np.zeros(dshape, dtype=np.complex128)
        dSdHx = 0.25 * dx * dy * np.conj(Ey)
        dSdHy = -0.25 * dx * dy * np.conj(Ex)
        dSdHz = np.zeros(dshape, dtype=np.complex128)

        dPdEx = -1 * f * dSdEx / Psrc**2
        dPdEy = -1 * f * dSdEy / Psrc**2
        dPdEz = -1 * f * dSdEz / Psrc**2
        dPdHx = -1 * f * dSdHx / Psrc**2
        dPdHy = -1 * f * dSdHy / Psrc**2
        dPdHz = -1 * f * dSdHz / Psrc**2

        dPdEH = interpolated_dFdx_3D(sim, z1, dPdEx, dPdEy, dPdEz,
                                     dPdHx, dPdHy, dPdHz)

        adj_sources.append(dPdEH[1:]); adj_domains.append(dPdEH[0])

    # clean up
    del Ex; del Ey; del Hx; del Hy

    # calculate dFdx for zmin
    dshape = z2.shape
    Ex = sim.get_field_interp(FieldComponent.Ex, domain=z2)
    Ey = sim.get_field_interp(FieldComponent.Ey, domain=z2)
    Hx = sim.get_field_interp(FieldComponent.Hx, domain=z2)
    Hy = sim.get_field_interp(FieldComponent.Hy, domain=z2)

    if(NOT_PARALLEL):
        dSdEx = 0.25 * dx * dy * np.conj(Hy)
        dSdEy = -0.25 * dx * dy * np.conj(Hx)
        dSdEz = np.zeros(dshape, dtype=np.complex128)
        dSdHx = -0.25 * dx * dy * np.conj(Ey)
        dSdHy = 0.25 * dx * dy * np.conj(Ex)
        dSdHz = np.zeros(dshape, dtype=np.complex128)

        dPdEx = -1 * f * dSdEx / Psrc**2
        dPdEy = -1 * f * dSdEy / Psrc**2
        dPdEz = -1 * f * dSdEz / Psrc**2
        dPdHx = -1 * f * dSdHx / Psrc**2
        dPdHy = -1 * f * dSdHy / Psrc**2
        dPdHz = -1 * f * dSdHz / Psrc**2

        dPdEH = interpolated_dFdx_3D(sim, z2, dPdEx, dPdEy, dPdEz,
                                     dPdHx, dPdHy, dPdHz)

        adj_sources.append(dPdEH[1:]); adj_domains.append(dPdEH[0])

    # clean up
    del Ex; del Ey; del Hx; del Hy

    return [adj_sources, adj_domains]

def power_norm_dFdx(sim, f, dfdA1, dfdA2, dfdA3):
    if(type(sim) == fdfd.FDFD_TM):
        power_norm_dFdx_TM(sim, f, dfdAi, dfdA2, dfdA3)
    elif(type(sim) == fdfd.FDFD_TE):
        power_norm_dFdx_TE(sim, f, dfdAi, dfdA2, dfdA3)
