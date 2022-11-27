from typing import Tuple

import pylab as pl
from photutils import CircularAperture, aperture_photometry


def measure_kernel(target_psf: pl.ndarray, result_psf: pl.ndarray, kernel: pl.ndarray) -> Tuple[float, float]:
    """Calculate two measurements of the matching kernel defined by Aniano+2011.

    Args:
        target_psf (image or 2D array): the targeted PSF to match.
        result_psf (image or 2D array): the resulting PSF after PSF matching.
        kernel (image or 2D array): the kernel used to do the PSF matching.

    Returns:
        D (float): D parameter in Aniano+2011, describes the accuracy in redistribution of PSF power.
        W_minus (float): W- parameter in Aniano+2011, describes the negative values of the kernel.
    """

    D: float = pl.sum(abs(target_psf - result_psf))
    W_minus: float = pl.sum(abs(kernel) - kernel) / 2
    return D, W_minus


def get_psf_properties(psf, napers=100, ap_max=None):
    """Obtain two PSF properties: (1) the enclosed power as a function of apertures,
    and (2) the normalized r*PSF(r) that indicates the power per unit radius.

    Args:
        psf (image or 2D array): the PSF to be calculated.
        napers (float, optional): the number of apertures used to compute the parameters.
            Default is 100.
        ap_max (float, optional): the maximum radius of the apertures. When ap_max is None (default),
            the half width of the PSF is adopted.

    Returns:
        aper_rads (array): the radii of the apertures.
        int_flux (array): the enclosed power of the input PSF.
        rpsf_norm (array): the normalized r*PSF(r) of the input PSF.
    """

    shape = psf.shape
    xcent = shape[0] / 2 - 1
    ycent = shape[1] / 2 - 1
    if ap_max is None:
        ap_max = int(min(shape) / 2)
    positions = pl.transpose((xcent, ycent))
    aper_rads = pl.linspace(0.1, ap_max, napers)
    apers = []
    for i in range(napers):
        apers.append(CircularAperture(positions, r=aper_rads[i]))
    tab_psf = aperture_photometry(psf, apers)
    int_flux = pl.array(list(tab_psf[0])[3:])
    den_flux = pl.zeros(napers - 1)
    for i in range(napers - 1):
        # average flux within the bin
        den_flux[i] = (int_flux[i + 1] - int_flux[i]) / (pl.pi * (aper_rads[i + 1]**2 - aper_rads[i]**2))
    rpsf_norm = den_flux * aper_rads[:-1] / max(den_flux * aper_rads[:-1])
    return aper_rads, int_flux, rpsf_norm


def check_psfmatch(source_psf, target_psf, result_psf, kernel, ap_max=None, psf_labels=None, isret=False, issave=False, figname=None):
    """Plot two figures to compare the source and resulting PSF and thus examine the performance of the PSF matching:
        (1) the enclosed power as a function of apertures,
        and (2) the normalized r*PSF(r) that indicates the power per unit radius.
        Ref: Aniano+2011 (http://adsabs.harvard.edu/abs/2011PASP..123.1218A)

    Args:
        source_psf (image or 2D array): the source PSF that to be convolved with the kernel.
        target_psf (image or 2D array): the targeted PSF to match.
        result_psf (image or 2D array): the resulting PSF after PSF matching.
        kernel (image or 2D array): the kernel used to do the PSF matching.
            All the PSFs and kernel should have the same shape.
        ap_max (float, optional): the maximum radius of the apertures. When ap_max is None (default),
            the half width of the PSF is adopted.
        psf_labels (list, optional): list of three strings that describes source_psf, target_psf, and result_psf, respectively.
            When it is None (default), [``source_psf'', ``target_psf'', ``result_psf''] will be used.
        isret (bool, optional): If ``True'', two measurements described the kernel performance (D and W-) will return.
            Default is ``False''.
        issave (bool, optional): If ``True'', the resulting figure will be saved as PDF file named ``figname''.
            Default is ``False''.
        figname (str, optional): the name of the saved figure, use only when ``issave=True''.

    Returns:
        D (float, optional): D parameter in Aniano+2011, describes the accuracy in redistribution of PSF power.
            Return when ``isret=True''.
        W_minus (float, optional): W- parameter in Aniano+2011, describes the negative values of the kernel.
            Return when ``isret=True''.

    Author: ZSLIN (zesenlin@ustc.edu.cn)
    """

    pl.figure(figsize=(10, 5))
    ax1 = pl.subplot2grid(shape=(4, 2), loc=(0, 0), rowspan=4, colspan=1)
    ax2 = pl.subplot2grid(shape=(4, 2), loc=(1, 1), rowspan=3, colspan=1)
    ax3 = pl.subplot2grid(shape=(4, 2), loc=(0, 1), rowspan=1, colspan=1)

    if psf_labels is None:
        psf_labels = ['source_psf', 'target_psf', 'result_psf']
    napers = 100
    aper_rads, int_flux1, rpsf_norm1 = get_psf_properties(source_psf, napers, ap_max)
    aper_rads, int_flux2, rpsf_norm2 = get_psf_properties(target_psf, napers, ap_max)
    aper_rads, int_flux3, rpsf_norm3 = get_psf_properties(result_psf, napers, ap_max)
    aper_rads, int_flux4, rkernel_norm = get_psf_properties(kernel, napers, ap_max)
    # plot the integrated power
    ax1.plot(aper_rads, int_flux1, 'b-.', label=psf_labels[0], zorder=2)
    ax1.plot(aper_rads, int_flux2, 'k-', label=psf_labels[1], zorder=2)
    ax1.plot(aper_rads, int_flux3, 'r--', label=psf_labels[2], zorder=2)
    ax1.hlines(0.5, 0, aper_rads[-1], colors='k', lw=1, linestyles='dashed', zorder=1)
    ax1.hlines(0.8, 0, aper_rads[-1], colors='k', lw=1, linestyles='dashed', zorder=1)
    ax1.hlines(1., 0, aper_rads[-1], colors='k', lw=1, linestyles='dashed', zorder=1)
    D, W_minus = measure_kernel(target_psf, result_psf, kernel)
    str_kernel_measure = r'$D=' + str(round(D, 2)) + ', W_{-}=' + str(round(W_minus, 2)) + '$'
    ax1.text(0.5, 0.1, str_kernel_measure, fontsize=15, horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes)
    ax1.set_xlabel('$r$[pixels]')
    ax1.set_ylabel('Enclosed power')
    ax1.set_xlim(0, aper_rads[-1])
    ax1.set_ylim(0, 1.05)

    # plot the profile
    ax2.plot(aper_rads[:-1], rpsf_norm1, 'b-.', label=psf_labels[0], zorder=2)
    ax2.plot(aper_rads[:-1], rpsf_norm2, 'k-', label=psf_labels[1], zorder=2)
    ax2.plot(aper_rads[:-1], rpsf_norm3, 'r--', label=psf_labels[2], zorder=2)
    ax2.plot(aper_rads[:-1], rkernel_norm, 'm:', label='kernel', zorder=2)
    ax2.hlines(0, 0, aper_rads[-1], colors='k', lw=1, linestyles='solid', zorder=1)
    ax2.set_xlabel('$r$[pixels]')
    ax2.set_ylabel(r'$r\times \mathrm{PSF}(r)/\mathrm{max}(r\times \mathrm{PSF}(r))$')
    ax2.set_xlim(0, aper_rads[-1])
    ax2.set_ylim(-0.2, 1.05)
    ax2.legend(framealpha=0.)

    # plot the differences
    ax3.hlines(0, 0, aper_rads[-1], colors='k', lw=1, linestyles='solid', zorder=1)
    ax3.plot(aper_rads[:-1], rpsf_norm2 - rpsf_norm3, 'r--', label=psf_labels[1] + '-' + psf_labels[2], zorder=2)
    ax3.set_xticklabels([])
    ax3.set_ylabel('Difference')
    ax3.set_xlim(0, aper_rads[-1])
    ax3.set_ylim(-0.5, 0.5)
    ax3.legend(framealpha=0.)
    if issave:
        pl.savefig(figname, format='pdf', bbox_inches='tight', pad_inches=0.1)
    if isret:
        return D, W_minus
