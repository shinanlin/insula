"""Native HGA filterbank with approved analytic-signal orthogonalization.

Hipp et al. (2012), doi:10.1038/nn.3101. Each existing Gaussian subband
is projected into quadrature with the reference BEFORE envelope extraction.
This removes instantaneous in-phase signal, not every possible shared source.
"""
import numpy as np
from scipy.fft import ifft
from ieeg.timefreq.hilbert import filterbank_hilbert_first_half_wrapper


def analytic_filterbank(data, sfreq):
    """Return time x channel x band, using native IEEG kernel/float precision."""
    x = np.asarray(data,dtype='float32').T.copy()
    if not np.isfinite(x).all():
        raise ValueError('Native voltage contains nonfinite samples')
    xf, freqs, centers, n, sds, h = filterbank_hilbert_first_half_wrapper(
        x,int(sfreq),70.,200.,1/7)
    kernel = np.zeros((n,len(centers)),dtype='complex64')
    k = freqs[:,None]-centers[None,:]
    kernel[:len(freqs)] = np.exp(-.5*(k/sds)**2).astype('complex64')
    kernel[len(freqs):] = kernel[1:(n+1)//2][::-1]
    kernel[0] = 0
    kernel *= h
    analytic = ifft(xf[:,:,None]*kernel[:,None,:],n,axis=0).astype('complex64')
    return analytic,centers


def directional_quadrature(target, reference):
    """Complex orthogonal signal; its envelope is |Im(A B*/|B|)|."""
    magnitude = np.abs(reference)
    unit = np.divide(reference,magnitude,out=np.zeros_like(reference),where=magnitude>0)
    # With an exactly zero reference there is no direction to remove.
    # Mark it unobserved rather than silently inventing residual activity.
    result = 1j*unit*np.imag(target*np.conj(unit))
    result[magnitude == 0] = np.nan
    return result


def pair_envelopes(data, sfreq):
    """Input trials x 2 x samples; output trials x 4 x samples.

    Channel order: AIC before, AIC perpendicular FO, FO before, FO perpendicular AIC.
    Sum subband envelopes exactly as ieeg.timefreq.gamma.extract does.
    """
    result = np.empty((len(data),4,data.shape[-1]),dtype='float64')
    zero_samples = 0
    for i,trial in enumerate(data):
        analytic,centers = analytic_filterbank(trial,sfreq)
        a,b = analytic[:,0],analytic[:,1]
        ap = directional_quadrature(a,b)
        bp = directional_quadrature(b,a)
        zero_samples += int((np.abs(a)==0).sum()+(np.abs(b)==0).sum())
        result[i] = np.stack([np.abs(a).sum(-1),np.abs(ap).sum(-1),
                              np.abs(b).sum(-1),np.abs(bp).sum(-1)])
    return result,dict(filter_centers_hz=centers.tolist(),
                       exact_zero_reference_subband_samples=zero_samples)
