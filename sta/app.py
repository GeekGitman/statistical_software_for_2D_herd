#!/usr/bin/env python

import wx
import mainFrame
import time

class StaApp(wx.App):
    
    def readConfig(self,path="./config/app.conf"):
        # read configure document and save in self.conf
        fp=open(path,'r')
        self.conf={}
        for lines in fp.readlines():
            lines=lines.replace("\n","").split(":")
            self.conf[lines[0]]=lines[1]
        fp.close()

    def __init__(self,redirect=False,filename=None):
        self.readConfig()
        # log file initiate
        t=time.localtime(time.time())
        self.logfile=self.conf['Redirect_path']+'log_'+str(t[0])+str(t[1])+str(t[2])+'.dat'
        fp=open(self.logfile,'a')
        fp.write('\n'+self.logfile+str(t[3])+str(t[4])+'\n')
        fp.close()
        # app initiate
        wx.App.__init__(self,redirect,self.logfile)

    def OnInit(self):
        # Splash screen setting
        #bmp=wx.Image("./bitmap/splash.jpg").ConvertToBitmap()
        #wx.SplashScreen(bmp,wx.SPLASH_CENTER_ON_SCREEN|wx.SPLASH_TIMEOUT,0,None,-1)
        #wx.SafeYield()
        #time.sleep(2)
        # Top Frame setting
        TopFrame=mainFrame.topframe(parent=None,name=self.conf['appName'])
        TopFrame.Show(True)
        self.SetTopWindow(TopFrame)
        return True
