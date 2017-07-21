#!/usr/bin/env python
import numpy as np
from cal import calculate

import wx
import time
#import panel
import ControlPanel
#import figure
import Matfigure
import sys
import json
import re
import csv
import os
from scipy import spatial
from settingDialog import SetDialog

class topframe(wx.Frame):

    def split(self):
        # Split window into two parts
        self.sp=wx.SplitterWindow(self,size=self.GetSize())
        self.pup=Matfigure.MatFigurePanel(self.sp,self)
        self.pup.SetBackgroundColour("LIGHT GREY")
        self.pdown=ControlPanel.ControlPanel(self.sp,-1,self.pup,self)
        self.pdown.SetBackgroundColour("WHITE")
        self.splitpos=self.GetSize()[0]-220.      # position of split line
        #print self.splitpos
        self.sp.SplitHorizontally(self.pup,self.pdown,self.splitpos)

    def __init__(self,parent,name):
        time.sleep(0)
        self.dataInit()

        wx.Frame.__init__(self,parent,-1,name)
        self.Displaysize=wx.DisplaySize()  # get display size
        #self.SetSize((1366,768))  # Default:1366*768 resizable
        self.SetSize((800,720))#this has been changed!!!!!!!!!
        self.split()
        # menu bar
        self.setMenu()
        # status bar
        self.setStatusBar()
        # other
        self.Centre()
        self.data = []
        self.result = {}
        self.savepath = ''
        self.inputpath = ''
        self.importpath = ''
        #self.OnOpen(wx.EVT_MENU)

    def dataInit(self):
        self.neighborRadius=0.2# r = self.neighborRadius * commonR
        self.NumOfRadius=10  #dr = max_length/NumOfRadius
        self.deltaAngle=0.2# angle = self.deltaAngle * math.pi
        self.step=1
        self.quantity1='Order parameter'
        self.quantity2='Order parameter'
        self.interval=25# ms

    def setMenu(self):
        menuBar = wx.MenuBar()
        # menu 1
        filemenu = wx.Menu()
        opens = filemenu.Append(wx.ID_OPEN,"&Open")
        save = filemenu.Append(wx.ID_SAVE,"&Save")
        saveas = filemenu.Append(wx.ID_ANY,"S&aveas")
        filemenu.AppendSeparator()
        saveall = filemenu.Append(wx.ID_ANY,'S&aveAll')
        save_all_figure =filemenu.Append(wx.ID_ANY,'S&aveAllFigure')
        imports = filemenu.Append(wx.ID_ANY,"I&mport")
        quit = filemenu.Append(wx.ID_EXIT,"&Quit")
        menuBar.Append(filemenu,"&Data")
        # menu 2
        editmenu = wx.Menu()
        stop=editmenu.Append(wx.ID_STOP,'&Stop')
        start=editmenu.Append(wx.ID_ANY,'S&tart')
        setting = editmenu.Append(wx.ID_ANY,"S&ettings")
        menuBar.Append(editmenu,"&Edit")
        # bind handler
        self.Bind(wx.EVT_MENU,self.OnOpen,opens)
        self.Bind(wx.EVT_MENU,self.OnSave,save)
        self.Bind(wx.EVT_MENU,self.OnSaveas,saveas)
        self.Bind(wx.EVT_MENU, self.OnSaveall, saveall)
        self.Bind(wx.EVT_MENU,self.OnSaveAllFigure,save_all_figure)
        self.Bind(wx.EVT_MENU,self.OnImport,imports)
        self.Bind(wx.EVT_MENU,self.OnExit,quit)
        self.Bind(wx.EVT_MENU,self.OnStop,stop)
        self.Bind(wx.EVT_MENU,self.OnStart,start)
        self.Bind(wx.EVT_MENU,self.OnSet,setting)
        # show menu
        self.SetMenuBar(menuBar)

    def setStatusBar(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText('Ready.')

    def OnOpen(self,evt):   #I made some changes
        print '#open#'
        OpenFile = wx.FileDialog(self,"Open data files","","","data files (*.csv)|*.csv",wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if OpenFile.ShowModal() == wx.ID_CANCEL:
            print 'Cancel'
            return
        self.inputpath = OpenFile.GetPath()
        print 'path:',self.inputpath

        #self.inputpath='E:\summer_research_in2016\sheep\sta\data\drinking.csv'
        # import data

    def OnSave(self,evt):
        print '#Save#'
        if self.savepath=='':
            self.savepath = self.importpath
        if self.savepath =='':
            print 'save failure.'
            wx.MessageBox('No path given.Please use Saveas.','alert',wx.OK)
            return
        self.saveas()

    def OnSaveas(self,evt):
        print '#Saveas#'
        SaveFile = wx.FileDialog(self,"Save result files","","sample.json","result files (*.json)|*.json",wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if SaveFile.ShowModal() == wx.ID_CANCEL:
            print 'Cancel'
            return
        if re.search('.json',SaveFile.GetPath())==None:
            wx.MessageBox('Not .json files.','error',wx.OK)
            print 'failure'
            return
        self.savepath = SaveFile.GetPath()
        self.saveas()

    def saveas(self):
        if self.savepath=='':
            return
        try:
            with open(self.savepath,'w') as f:
                json.dump(self.result,f)
        except:
            info = sys.exc_info()
            print info[0],':',info[1]
        f.close()

    def OnSaveall(self,evt):
        dialog = wx.DirDialog(self, "Choose a directory", os.getcwd())
        if dialog.ShowModal() == wx.ID_CANCEL:
            print 'Cancel'
            return
        path=dialog.GetPath()
        print path
        try:
            conf=self.pdown.ListContent
            Info=self.pdown.ListInfo
            for k in conf.keys():
                a=k.replace(' ','')
                with open(path+"/"+a+'.json', 'w') as f:
                    result = eval('calculate.' + conf[k])()
                    json.dump(Info[k],f)
                    for result_i in result:
                        f.write('\n')
                        json.dump(result_i, f)
                    f.close()
        except:
            info = sys.exc_info()
            print info[0], ':', info[1]
            wx.MessageBox('Could not write to this.', 'alert', wx.OK)

    def OnSaveAllFigure(self,evt):
        dialog = wx.DirDialog(self, "Choose a directory", os.getcwd())
        if dialog.ShowModal() == wx.ID_CANCEL:
            print 'Cancel'
            return
        path = dialog.GetPath()
        print path
        try:
            conf = self.pdown.ListContent
            Info = self.pdown.ListInfo
            for k in conf.keys():
                    fig = path+'\\'+k.replace(' ', '')
                    result = eval('calculate.' + conf[k])()
                    if Info[k]['dataType'] == 'Lines':
                        self.pup.plotLines(Info[k],result,savefigure=fig+'.png')
                    else:
                        if not os.path.isdir(fig):
                            os.makedirs(fig)
                        eval('self.pup.plot'+Info[k]['dataType'])(Info[k],result,savefigure=fig)
        except:
            info = sys.exc_info()
            print info[0], ':', info[1]
            wx.MessageBox('Could not write to this.', 'alert', wx.OK)

    def OnImport(self,evt):
        print '#Import#'
        ImportFile = wx.FileDialog(self,"Import result files","","","result files (*.json)|*.json",wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if ImportFile.ShowModal() == wx.ID_CANCEL:
            print 'Cancel'
            return
        self.importpath = ImportFile.GetPath()
        self.pdown.OnReHandler(wx.EVT_BUTTON)

    def OnExit(self,evt):
        self.Close()

    def OnStop(self,evt):
        print '#stop#'
        self.pup.timer.stop()

    def OnStart(self, evt):
        print '#start#'
        self.pup.timer.start()

    def OnSet(self,evt):
        print '#Set#'
        sdg=SetDialog(self,'setting')
        sdg.Show()
