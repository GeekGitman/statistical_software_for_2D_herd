#!/usr/bin/env python

import wx
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MatFigurePanel(wx.Panel):

    def __init__(self, parent ,datainit):
        wx.Panel.__init__(self, parent=parent, id=-1)
        self.parent = datainit
        self.Figure = plt.Figure()
        self.axes = self.Figure.add_axes([0.1, 0.1, 0.85, 0.85])
        self.FigureCanvas = FigureCanvas(self, -1, self.Figure)
        self.timer=self.FigureCanvas.new_timer(interval=25)
        self.NavigationToolbar = NavigationToolbar(self.FigureCanvas)
        self.SubBoxSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SubBoxSizer.Add(self.NavigationToolbar, proportion=0, border=2, flag=wx.ALL | wx.EXPAND)
        self.TopBoxSizer = wx.BoxSizer(wx.VERTICAL)
        self.TopBoxSizer.Add(self.SubBoxSizer, proportion=-1, border=2, flag=wx.ALL | wx.EXPAND)
        self.TopBoxSizer.Add(self.FigureCanvas, proportion=-10, border=2, flag=wx.ALL | wx.EXPAND)
        self.SetSizer(self.TopBoxSizer)

    def plotLines(self, Information, data, error_list=None, savefigure=False):
        self.Figure.clear()
        self.timer.stop()
        self.axes=self.Figure.add_axes([0.1, 0.1, 0.85, 0.85],polar=Information['polar'])
        if error_list == None:
            if Information['legend']:
                self.axes.errorbar(data[0], data[1], xerr=None, yerr=None, fmt='o-',
                label=Information['legend'], lw=1)  # frame should be changed
            else:
                self.axes.errorbar(data[0], data[1], xerr=None, yerr=None, fmt='o-', lw=1)
        else:
            if Information['legend']:
                self.axes.errorbar(data[0], data[1], xerr=None, yerr=error_list, fmt='o-',
                label=Information['legend'], lw=1)  # frame should be changed
            else:
                self.axes.errorbar(data[0], data[1], xerr=None, yerr=error_list, fmt='o-', lw=1)
        if Information['legend']:
            self.axes.legend()
        self.axes.set_xlabel(Information['xlabel'])
        self.axes.set_ylabel(Information['ylabel'])
        parent=self.Parent.Parent
        if Information['Title']=='Correlation-Tau':
            self.axes.set_title(Information['Title']+' between '+parent.quantity1+' and '+parent.quantity2)
        else:
            self.axes.set_title(Information['Title'])
        if savefigure:
            self.Figure.savefig(savefigure, transparent=True)
        if not savefigure:
            self.FigureCanvas.draw()

    def plotMultipleLine(self, Information, data, error_list=None, savefigure=False):
        self.Figure.clear()    
        self.timer.stop()
        self.axes=self.Figure.add_axes([0.1, 0.1, 0.85, 0.85],polar=Information['polar'])
        dat=[]
        for index,dat in enumerate(data):
            c = np.arange(0,0.9+0.45/dat[2],0.9/dat[2])
            if error_list == None:
                if Information['legend']:
                    self.axes.errorbar(dat[0], dat[1], xerr=None, yerr=None, fmt='o-r', color=str(c[index]),
                    label=Information['legend']+str(index*self.parent.step), lw=1)  # frame should be changed
                else:
                    self.axes.errorbar(dat[0], dat[1], xerr=None, yerr=None, fmt='o-r'+c, lw=1)
            else:
                if Information['legend']:
                    self.axes.errorbar(dat[0], dat[1], xerr=None, yerr=error_list, fmt='o-r', color=str(c[index]),
                    label=Information['legend']+str(index*self.parent.step), lw=1)  # frame should be changed
                else:
                    self.axes.errorbar(dat[0], dat[1], xerr=None, yerr=error_list, fmt='o-r'+c, lw=1)
        if Information['legend']:
            self.axes.legend()
        self.axes.set_xlabel(Information['xlabel'])
        self.axes.set_ylabel(Information['ylabel'])
        parent=self.Parent.Parent
        if Information['Title']=='Correlation-deltaFrame':
            self.axes.set_title(Information['Title']+' between '+parent.quantity1+' and '+parent.quantity2)
        else:
            self.axes.set_title(Information['Title'])
        if savefigure:
            self.Figure.savefig(savefigure, transparent=True)
        if not savefigure:
            self.FigureCanvas.draw()
        

    def plotMoveLine(self,Information,data,error_list=None,savefigure=False):
        if not savefigure:
            self.Figure.clear()
            self.timer.stop()
            self.axes = self.Figure.add_axes([0.1, 0.1, 0.85, 0.85], polar=Information['polar'])
            a = np.array(data.next())
            lines=[]
            for ii,data_item in enumerate(a):
                if error_list==None:
                    if Information['legend']:
                        line,=self.axes.plot(data_item[0],data_item[1],fmt='o-',label=Information['legend'][ii],lw=1,animated=True)#frame should be changed
                    else:
                        line, =self.axes.plot(data_item[0], data_item[1], 'o-', lw=1,animated=True)
                else:
                    if Information['legend']:
                        line, =self.axes.errorbar(data_item[0], data_item[1], xerr=None, yerr=error_list[ii], fmt='o-',label=Information['legend'][ii], lw=1,animated=True)  # frame should be changed
                    else:
                        line, =self.axes.errorbar(data_item[0], data_item[1], xerr=None, yerr=error_list[ii], fmt='o-', lw=1,animated=True)
                lines.append(line)
            if Information['legend']:
                self.axes.legend()
            self.axes.set_xlabel(Information['xlabel'])
            self.axes.set_ylabel(Information['ylabel'])
            self.axes.set_title(Information['Title'])
            #self.axes.set_xlim(*Information['xlim'])
            self.FigureCanvas.draw()
            background = self.FigureCanvas.copy_from_bbox(self.axes.bbox)
            def update_data(lines,data):
                self.FigureCanvas.restore_region(background)
                try:
                    b = np.array(data.next())
                except:
                    print 'out'
                    self.timer.stop()
                    #self.timer.remove_callback(self.timer.callbacks)
                    return
                for i, a in enumerate(b):
                    lines[i].set_xdata(a[0])
                    lines[i].set_ydata(a[1])
                    self.axes.draw_artist(lines[i])
                change = False
                dataXm=np.max(a[0])
                dataYm=np.max(a[1])
                while True:
                    xmin, xmax = self.axes.get_xlim()
                    ymin, ymax = self.axes.get_ylim()
                    if dataXm>xmax:
                        self.axes.set_xlim(xmin,1.2*xmax)
                        change=True
                    elif dataYm > ymax:
                        self.axes.set_ylim(ymin, 1.2 * ymax)
                        change=True
                    else:
                        if change:
                            self.FigureCanvas.draw()
                        break
                self.FigureCanvas.blit(self.axes.bbox)

            self.timer = self.FigureCanvas.new_timer(interval=25)
            self.timer.add_callback(update_data, lines,data)
            self.timer.start()

        if  savefigure:
            for i, a in enumerate(data):
                plt.clf()
                axes=plt.axes(polar=Information['polar'])
                for ii, data_item in enumerate(a):
                    if error_list == None:
                        if Information['legend']:
                            axes.plot(data_item[0], data_item[1], xerr=None, yerr=None, fmt='o-',
                                      label=Information['legend'][ii], lw=1)  # frame should be changed
                        else:
                            axes.plot(data_item[0], data_item[1], lw=1)
                    else:
                        if Information['legend']:
                            axes.errorbar(data_item[0], data_item[1], xerr=None, yerr=error_list[ii], fmt='o-',
                                          label=Information['legend'][ii], lw=1)  # frame should be changed
                        else:
                            axes.errorbar(data_item[0], data_item[1], xerr=None, yerr=error_list[ii], fmt='o-', lw=1)

                if Information['legend']:
                    axes.legend()
                axes.set_xlabel(Information['xlabel'])
                axes.set_ylabel(Information['ylabel'])
                axes.set_title(Information['Title'])
                # axes.set_xlim(*Information['xlim'])
                plt.savefig(savefigure + '\\frame=' + str(i) + '.png')

    def plotScatter(self,Information,data,savefigure=False):
        # draw velocities
        cmap = matplotlib.cm.jet
        norm = matplotlib.colors.Normalize()
        self.timer.stop()
        if not savefigure:
            self.Figure.clear()
            self.axes = self.Figure.add_axes([0.1, 0.1, 0.85, 0.85])
            value = np.array(data.next())
            pic = self.axes.scatter(value[:, 0], value[:, 1], c=value[:, 2],animated=True,norm=norm,cmap=cmap)
            self.axes.set_xlabel(Information['xlabel'])
            self.axes.set_ylabel(Information['ylabel'])
            self.axes.set_title(Information['Title'])
            # self.axes.set_xlim(*Information['xlim'])
            self.Figure.colorbar(pic)
            self.FigureCanvas.draw()
            background=self.FigureCanvas.copy_from_bbox(self.axes.bbox)
            def update_data(pic):
                try:
                    a = np.array(data.next())
                except:
                    self.timer.stop()
                    return
                points=a[:,0:2]
                pic.set_offsets(points)
                pic.set_array(a[:,2])
                dataMax=np.max(points,0)
                dataMin=np.min(points,0)
                change=False
                while True:
                    xmin, xmax = self.axes.get_xlim()
                    ymin, ymax = self.axes.get_ylim()
                    if dataMax[0] > xmax:
                        self.axes.set_xlim(xmin, xmax+   0.2 * (xmax-xmin))
                        change = True
                    elif dataMax[1]> ymax:
                        self.axes.set_ylim(ymin, ymax +  0.2 * (ymax-ymin))
                        change = True
                    elif dataMin[0] < xmin:
                        self.axes.set_xlim(xmin - 0.2 * (xmax - xmin),xmax)
                        change = True
                    elif dataMin[1] < ymin:
                        self.axes.set_ylim(ymin - 0.2 * (ymax - ymin),ymax)
                        change = True
                    else:
                        if change:
                            self.FigureCanvas.draw()
                        break
                #pic._sizes=None
                self.FigureCanvas.restore_region(background)
                self.axes.draw_artist(pic)
                self.FigureCanvas.blit(self.axes.bbox)

            self.timer = self.FigureCanvas.new_timer(interval=25)
            self.timer.add_callback(update_data, pic)
            self.timer.start()
        if savefigure:
            for i,data_frame in enumerate(data):
                plt.clf()
                value = np.array(data_frame)
                pic=plt.scatter(value[:, 0], value[:, 1], c=value[:, 2],norm=norm,cmap=cmap)
                plt.xlabel(Information['xlabel'])
                plt.ylabel(Information['ylabel'])
                plt.title(Information['Title'])
                plt.colorbar(pic)
                plt.savefig(savefigure+'\\frame='+str(i)+'.png')

    def plotTrace(self,Information,data,savefigure=False):
        cmap = matplotlib.cm.jet
        norm = matplotlib.colors.Normalize()
        self.timer.stop()
        if not savefigure:
            self.Figure.clear()
            self.axes = self.Figure.add_axes([0.1, 0.1, 0.85, 0.85])
            qv=[]
            cbar=[]
            for ii,a in enumerate(data.next()):
                a=np.array(a)
                xmin,ymin = np.min(a[:,0:2],0)
                xmax,ymax = np.max(a[:,0:2],0)
                #if np.size(a,1)==4:
                qv.append(self.axes.quiver(a[:,0],a[:,1],a[:,2],a[:,3],np.hypot(a[:,2],a[:,3]),animated=True,pivot='tail',scale=max((np.hypot(a[:,2],a[:,3])))/(0.1*float(xmax-xmin)),units='x'))
                cbar=self.Figure.colorbar(qv[ii])
                #elif np.size(a,1)==5:
                #    qv.append(self.axes.quiver(a[:,0],a[:,1],a[:,2],a[:,3],np.hypot(a[:,2],a[:,3]),animated=True,label=ii,lw=1,norm=norm,cmap=cmap))
                #    self.Figure.colorbar(qv[ii])  # how to make the color standard the same
                #else:
                #    print 'error'
                #    return

            self.axes.set_xlabel(Information['xlabel'])
            self.axes.set_ylabel(Information['ylabel'])
            self.axes.set_title(Information['Title'])
            self.axes.legend()
            self.FigureCanvas.draw()
            background=self.FigureCanvas.copy_from_bbox(self.axes.bbox)
            #cbar=self.Figure.colorbar(qv[0])
            def update_data(qv,cbar):
                self.FigureCanvas.restore_region(background)
                try:
                    b=np.array(data.next())
                except:
                    self.timer.stop()
                    return
                for i,a in enumerate(b):
                    points=a[:,0:2]
                    qv[i].set_offsets(points)
                    #if np.size(a,1)==4:
                    qv[i].set_UVC(a[:,2],a[:,3],np.hypot(a[:,2],a[:,3]))
                    cbar.update_normal(qv[i])
                    #elif np.size(a,1)==5:
                    #    qv[i].set_UVC(a[:,2],a[:,3],np.hypot(a[:,2],a[:,3]))
                    #else:
                    #    print 'error'
                    self.axes.draw_artist(qv[i])
                    change = False
                    dataMax = np.max(points, 0)
                    dataMin = np.min(points, 0)
                    while True:
                        xmin, xmax = self.axes.get_xlim()
                        ymin, ymax = self.axes.get_ylim()
                        if dataMax[0] > xmax:
                            self.axes.set_xlim(xmin, xmax + 0.2 * (xmax - xmin))
                            change = True
                        elif dataMax[1] > ymax:
                            self.axes.set_ylim(ymin, ymax + 0.2 * (ymax - ymin))
                            change = True
                        elif dataMin[0] < xmin:
                            self.axes.set_xlim(xmin - 0.2 * (xmax - xmin),xmax)
                            change = True
                        elif dataMin[1] < ymin:
                            self.axes.set_ylim( ymin - 0.2 * (ymax - ymin),ymax)
                            change = True
                        else:
                            if change:
                                self.FigureCanvas.draw()
                            break
                self.FigureCanvas.blit(self.axes.bbox)
            self.timer = self.FigureCanvas.new_timer(interval=self.Parent.Parent.interval)
            self.timer.add_callback(update_data, qv, cbar)
            self.timer.start()

            '''


            def update_qv(i):
                for ii, data_item in enumerate(data):
                    a = np.array(data_item[i])
                    qv[ii].set_offsets(a[:, 0:2])
                    qv[ii].set_UVC(a[:, 2], a[:, 3])
                if savefigure:
                    self.Figure.savefig(savefigure+'/frame='+str(i)+'.jpg')
                return qv


            ani=FuncAnimation(self.Figure,update_qv,blit=True,interval=25,frames=l,repeat=False)
            self.FigureCanvas.draw()
            '''
        if savefigure:
            for i,data_frame in enumerate(data):
                plt.clf()
                qv = []
                for ii, data_item in enumerate(data_frame):
                    a = np.array(data_item)
                    if np.size(a, 1) == 4:
                        qv.append(plt.quiver(a[:, 0], a[:, 1], a[:, 2], a[:, 3], 1,  label=ii, lw=0.1,norm=norm,cmap=cmap))
                    elif np.size(a, 1) == 5:
                        qv.append(plt.quiver(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], label=ii, lw=0.1,norm=norm,cmap=cmap))
                        plt.colorbar(qv[ii])  # how to make the color standard the same
                    else:
                        print 'error'
                        return
                plt.savefig(savefigure+'\\frame='+str(i)+'.png')

    '''
    def plotPicture(self,Information,data):
        self.Figure.clear()
        axes = self.Figure.add_axes([0.1, 0.1, 0.85, 0.85])
        pic=axes.imshow(data[0])
        self.Figure.colorbar(pic)
        self.FigureCanvas.draw()
        '''
