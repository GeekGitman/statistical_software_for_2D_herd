#!/usr/bin/env python

import wx
import wx.lib.agw.balloontip as wla
import numpy as np
import json
import re
import sys
from cal import calculate
import threading
import time


class ControlPanel(wx.Panel):
    def ReadList(self, path="./config/List.conf"):
        fp = open(path, 'r')
        self.ListContent = {}
        for lines in fp.readlines():
            line = lines.replace("\n", "").split(":")
            self.ListContent[line[0]] = line[1]
        fp.close()
        f=open("./config/name.txt",'w')
        for line in self.ListContent.keys():
            f.write(line+'\n')
        f.close()

    def ReadInformation(self,path='./config/Information.json'):
        f=open(path,'r')
        self.ListInfo=json.load(f)

    def __init__(self, parent, ID, figure_panel, dataParent):
        wx.Panel.__init__(self, parent, ID, style=wx.SUNKEN_BORDER)
        self.figure = figure_panel
        self.parent = dataParent
        # List contents
        self.ReadList()
        self.ReadInformation()
        # initiate widgets holder
        # put widgets
        self.createListButton(parent)
        # initate class calculate. pass figure panel and topframe to it
        calculate.init(figure_panel, self.parent)

    def createListButton(self, parent):
        listcontent = self.ListContent.keys()
        label=wx.StaticText(self, -1, "Quantitative:" ,(30,10))
        self.choice=wx.Choice(self, -1,(130,10), choices=listcontent)
        self.plotbutton=wx.Button(self, -1, "Calculate and Plot",(400,10))
        self.savebutton=wx.Button(self, -1, "Save Result",(550,10))
        self.Bind(wx.EVT_BUTTON, self.OnClickPlotButton, self.plotbutton)
        self.Bind(wx.EVT_BUTTON, self.OnClickSaveButton, self.savebutton)
        self.ResultHandler = wx.Button(self, -1, "Plot Result", pos=(650, 10))
        self.Bind(wx.EVT_BUTTON, self.OnReHandler, self.ResultHandler)
        self.plotbutton.SetToolTip(wx.ToolTip('Calculate and plot the quantitative you choose.'))
        self.savebutton.SetToolTip(wx.ToolTip('Save the calculating result in a json file.'))
        self.ResultHandler.SetToolTip(wx.ToolTip('Import result (json file) and plot it.'))

    def result(self):
        if self.parent.importpath == '':
            wx.MessageBox('No import path', 'alert', wx.OK)
            return
        f = open(self.parent.importpath, 'r')
        Info = json.loads(f.readline())
        return Info,(json.loads(line.strip('\n')) for line in f)

    def OnReHandler(self, event):
            Info ,result=self.result()
            eval('self.figure.plot'+Info['dataType'])(Info,result)
            # print self.data
            self.parent.statusbar.SetStatusText('Ready.Data field:' + self.parent.inputpath + ' ;Result field:' + self.parent.importpath)
            #info = sys.exc_info()
            #print info[0], ':', info[1]
            #wx.MessageBox('Could not import this file.', 'alert', wx.OK)

    def OnClickPlotButton(self, event):
        func = self.ListContent[self.choice.GetString(self.choice.GetSelection())]
        # call calculation function
        t0 = time.clock()
        result = eval('calculate.' + func)()
        t1 = time.clock()
        print 'time:', t1 - t0
        # self.figure.plotLines(*result)
        info=self.ListInfo[self.choice.GetString(self.choice.GetSelection())]
        eval('self.figure.plot' + info['dataType'])(info,result)
        print 100

    def OnClickSaveButton(self, event):
        func = self.ListContent[self.choice.GetString(self.choice.GetSelection())]
        # call calculation function
        SaveFile = wx.FileDialog(self, "Save result files", "", self.choice.GetString(self.choice.GetSelection()).replace(' ','')+".json", "result files (*.json)|*.json",
                                 wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if SaveFile.ShowModal() == wx.ID_CANCEL:
            print 'Cancel'
            return
        if re.search('.json', SaveFile.GetPath()) == None:
            wx.MessageBox('Not .json files.', 'error', wx.OK)
            print 'failure'
            return
        savepath = SaveFile.GetPath()

        try:
            Info=self.ListInfo[self.choice.GetString(self.choice.GetSelection())]
            with open(savepath, 'w') as f:
                t0 = time.clock()
                result = eval('calculate.' + func)()
                t1 = time.clock()
                print 'time:', t1 - t0
                json.dump(Info, f)
                for result_i in result:
                    f.write('\n')
                    json.dump(result_i, f)
        except:
            info = sys.exc_info()
            print info[0], ':', info[1]
        f.close()

