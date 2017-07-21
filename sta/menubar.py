#!/usr/bin/env python
#NO USE???

import wx

class MenuBars(wx.MenuBar):
    
    def __init__(self,parent):
        self.parent = parent
        self.data = [['Data',[('Open',self.OnOpen),('Save',self.OnSave),('Saveas',self.OnSaveas),'s',('Import',self.OnImport),'s',('Exit',self.OnExit)]],['Figure',[('SaveFigure',self.OnSaveFigure),('Set',self.OnSet)]]]
        self.creatMenubar()

    def creatMenubar(self):
        self.mbar = wx.MenuBar()
        for menudata in self.data:
            menu = wx.Menu()
            for itemdata in menudata[1]:
                if itemdata=='s':
                    menu.AppendSeparator()
                    continue
                item = menu.Append(-1,itemdata[0])
                self.Bind(wx.EVT_MENU,itemdata[1],item)
            self.mbar.Append(menu,menudata[0])

    def OnOpen(self,evt):
        print '#open#'

    def OnSave(self,evt):
        print '#Save#'

    def OnSaveas(self,evt):
        print '#Saveas#'

    def OnImport(self,evt):
        print '#Import#'

    def OnExit(self,evt):
        self.parent.Close()

    def OnSaveFigure(self,evt):
        print '#savefigure#'

    def OnSet(self,evt):
        print '#Set#'

