import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator

import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

from itertools import cycle

from matplotlib import gridspec
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.1e'%x))
fmt = mticker.FuncFormatter(g)

#scientific_formatter = FuncFormatter(scientific)
#ax.yaxis.set_major_formatter(scientific_formatter)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

plt.rcParams.update({'backend' : 'Qt5Agg'})
plt.rcParams.update({'text.usetex' : True})

plt.rcParams.update({'font.size' : 11.0})
plt.rcParams.update({'axes.titlesize' : 11.0})  # Font size of title
plt.rcParams.update({'axes.titlepad'  : 11.0})
plt.rcParams.update({'axes.labelsize' : 11.0})  # Axes label sizes
plt.rcParams.update({'axes.labelpad'  : 11.0})
plt.rcParams.update({'xtick.labelsize': 11.0})
plt.rcParams.update({'ytick.labelsize': 11.0})

plt.rcParams.update({'axes.spines.left'  : True})
plt.rcParams.update({'axes.spines.right' : True})
plt.rcParams.update({'axes.spines.top'   : True})
plt.rcParams.update({'axes.spines.bottom': True})
plt.rcParams.update({'savefig.format'    : 'pdf'})
plt.rcParams.update({'savefig.bbox'      : 'tight'})
plt.rcParams.update({'savefig.pad_inches': 0.1})
plt.rcParams.update({'pdf.compression'   : 6})

allcolors = ['#377eb8', '#ff7f00', 'forestgreen', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

def beautify(ax, ttl=None, bb=None):
    try:
        len(ax)
    except:
        ax = [ax]
    for ai, aa in enumerate(ax):
        aa.grid(which='both', ls=':', color='lightgray', alpha=0.7)
        aa.tick_params(direction='in', which='both', top=True, right=True)
        #aa.ticklabel_format(axis='both', style='scientific', scilimits=[0.,0.])
        aa.xaxis.set_label_coords(0.5, -0.15)
        aa.yaxis.set_label_coords(-0.15, 0.5)
        aa.xaxis.label.set_color('k')
        aa.yaxis.label.set_color('k')
        aa.tick_params(axis='x', colors='k')
        aa.tick_params(axis='y', colors='k')
        aa.tick_params(direction='in', which='both', bottom=None, left=None, top=None, right=None)
        aa.spines['left'].set_color('k')
        aa.spines['right'].set_color('k')
        aa.spines['top'].set_color('k')
        aa.spines['bottom'].set_color('k')
        aa.legend(title=ttl, bbox_to_anchor=bb, frameon=False, handlelength=1.5, labelspacing=0.3, columnspacing=0.6)
    return ax


def plot_dmdz(ax, zs, ms, func, count=10, ls='-'):
    c1, c2, g1, g2 = 0, 0, 0, 0
    for mi, mm in enumerate(ms):
        if mm in ms[:count]:
            ax[0].plot(zs, func[:,mi], ls=ls, lw=1, label=('$m={:.2e}$'.format(mm) if mm in ms[:count//count] and ls=='-' else None), alpha=1-c1/count, color='k')
            c1 += 1
        elif mm in ms[-count:]:
            ax[0].plot(zs, func[:,mi], ls=ls, lw=1, label=('$m={:.2e}$'.format(mm) if mm in ms[-count//count:] and ls=='-' else None), alpha=c2/count, color='g')
            c2 += 1
    for zi, zz in enumerate(zs):
        if zz in zs[:count]:
            ax[1].plot(ms, func[zi,:], ls=ls, lw=1, label=('$z={:.2f}$'.format(zz) if zz in zs[:count//count] and ls=='-' else None), alpha=1-g1/count, color='r')
            g1 += 1
        elif zz in zs[-count:]:
            ax[1].plot(ms, func[zi,:], ls=ls, lw=1, label=('$z={:.2f}$'.format(zz) if zz in zs[-count//count:] and ls=='-' else None), alpha=g2/count, color='b')
            g2 += 1

    ax[0].set_xlabel(r'$z$')
    ax[1].set_xlabel(r'$m$')
    return ax

def plot_ucosth(ms, zs, angs, ucosth, prob, title, count=10):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    nMs, nZs = len(ms), len(zs)
    c1, c2, g1, g2 = 0, 0, 0, 0
    for mi, mm in enumerate(ms):
        if mm in ms[:count]:
            lab = lambda zi: (r'$m={:.1e}, z={:.1f}$'.format(mm, zs[zi]) if mi==0 else None)
            zi = 0
            ax[0].plot(angs[:,zi,mi], ucosth[:,zi,mi], label=lab(zi), alpha=c1/count, color='k')
            c1 += 1
            zi = nZs-1
            ax[0].plot(angs[:,zi,mi], ucosth[:,zi,mi], label=lab(zi), alpha=c2/count, color='r')
            c2 += 1
        elif mm in ms[-count:]:
            lab = lambda zi: (r'$m={:.1e}, z={:.1f}$'.format(mm, zs[zi]) if mi==len(ms)-1 else None)
            zi = 0
            ax[0].plot(angs[:,zi,mi], ucosth[:,zi,mi], label=lab(zi), alpha=1-g1/count, color='g')
            g1 += 1
            zi = nZs-1
            ax[0].plot(angs[:,zi,mi], ucosth[:,zi,mi], label=lab(zi), alpha=1-g2/count, color='b')
            g2 += 1

    c1, c2, g1, g2 = 0, 0, 0, 0
    for mi, mm in enumerate(ms):
        if mm in ms[:count]:
            lab = lambda zi: (r'$m={:.1e}, z={:.1f}$'.format(mm, zs[zi]) if mi==0 else None)
            zi = 0
            ax[1].plot(angs[:,zi,mi], ucosth[:,zi,mi]*prob[zi,mi], label=lab(zi), alpha=c1/count, color='k')
            c1 += 1
            zi = nZs-1
            ax[1].plot(angs[:,zi,mi], ucosth[:,zi,mi]*prob[zi,mi], label=lab(zi), alpha=c2/count, color='r')
            c2 += 1
        elif mm in ms[-count:]:
            lab = lambda zi: (r'$m={:.1e}, z={:.1f}$'.format(mm, zs[zi]) if mi==len(ms)-1 else None)
            zi = 0
            g1 += 1
            ax[1].plot(angs[:,zi,mi], ucosth[:,zi,mi]*prob[zi,mi], label=lab(zi), alpha=1-g1/count, color='g')
            zi = nZs-1
            ax[1].plot(angs[:,zi,mi], ucosth[:,zi,mi]*prob[zi,mi], label=lab(zi), alpha=1-g2/count, color='b')
            g2 += 1

    for axx in ax:
        axx.set_yscale('log'); axx.set_xscale('log')
        axx.set_ylabel(r'$u(\cos(\theta))$'); axx.set_xlabel(r'$\theta$')
        axx.legend(); axx.grid()
    return ax

def plot_sigma(ax, ms, zs, ells, ucosth, count=10, ls='-'):
    nMs, nZs = len(ms), len(zs)
    c1, c2, g1, g2 = 0, 0, 0, 0
    for mi, mm in enumerate(ms):
        if mm in ms[:count]:
            lab = lambda zi: (r'$m={:.2e}, z={:.2f}$'.format(mm, zs[zi]) if mi==0 and ls=='-' else None)
            zi = 0
            ax.plot(ells, ucosth[:,zi,mi], label=lab(zi), lw=1, ls=ls, alpha=1-c1/count, color='k')
            c1 += 1
            zi = nZs-1
            ax.plot(ells, ucosth[:,zi,mi], label=lab(zi), lw=1, ls=ls, alpha=1-c2/count, color='r')
            c2 += 1
        elif mm in ms[-count:]:
            lab = lambda zi: (r'$m={:.2e}, z={:.2f}$'.format(mm, zs[zi]) if mi==len(ms)-1 and ls=='-' else None)
            zi = 0
            ax.plot(ells, ucosth[:,zi,mi], label=lab(zi), lw=1, ls=ls, alpha=g1/count, color='g')
            g1 += 1
            zi = nZs-1
            ax.plot(ells, ucosth[:,zi,mi], label=lab(zi), lw=1, ls=ls, alpha=g2/count, color='b')
            g2 += 1

    for axx in [ax]:
        axx.set_xlabel(r'$\ell$')
        axx.set_ylabel(r'$C_{\ell}^{\alpha}$')
    return ax

#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].axes
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)

def annot_max(xmax, ymax, lab, col, xcap, mind, ax):
    text = lab
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec=col, lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60",color=col)
    kw = dict(xycoords='data',textcoords="data",
              arrowprops=arrowprops, bbox=bbox_props, ha="left", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax-0.01, ymax-5e6), **kw)
