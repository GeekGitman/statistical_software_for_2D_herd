import wx

#dialog for setting
class SetDialog(wx.Dialog):

    def __init__(self,parent,name):
        wx.Dialog.__init__(self,parent,-1,name)
        self.Items={}
        self.Buttons={}
        self.sizer=wx.GridSizer(8,2)
        self.addText('neighborRadius',value=parent.neighborRadius)
        self.addText('NumOfRadius',value=parent.NumOfRadius)
        self.addText('deltaAngle',value=parent.deltaAngle)
        self.addText('step of data',value=parent.step)
        self.addText('interval',value=parent.interval)

        self.addChoice()
        self.SetButton()
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.sizer.Fit(self)

    def addText(self,name,value):
        label=wx.StaticText(self,label=name+':')
        if name=='interval':
            sty=wx.TE_READONLY
        else:
            sty=wx.TE_LEFT  
        edit=wx.TextCtrl(self,value=str(value),style=sty)
        self.Items[name]=edit
        self.sizer.AddMany([label,edit])

    def addChoice(self):
        list=self.Parent.pdown.ListInfo
        list=[name for name in list.keys() if list[name]['dataType']=='Lines' and name!='Correlation']
        label1=wx.StaticText(self,-1,label='quantity1')
        label2 = wx.StaticText(self,-1, label='quantity2')
        self.choice1=wx.Choice(self,-1,choices=list)
        self.choice2=wx.Choice(self,-1,choices=list)
        self.sizer.AddMany([label1,self.choice1,label2,self.choice2])

    def SetButton(self):
        self.Buttons['OK'] =okButton= wx.Button(self,wx.ID_OK,'OK')
        self.Buttons['CANCEL'] =cancelButton= wx.Button(self,wx.ID_CLOSE,'CANCEL')
        self.Bind(wx.EVT_BUTTON, self.OnOK,okButton)
        self.Bind(wx.EVT_BUTTON, self.OnCANCEL, cancelButton)
        self.sizer.AddMany([self.Buttons['OK'], self.Buttons['CANCEL']])

    def OnOK(self,event):
        print '#OnOK!#'
        parent=self.Parent
        parent.neighborRadius=float(self.Items['neighborRadius'].GetValue())
        parent.NumOfRadius = float(self.Items['NumOfRadius'].GetValue())
        parent.deltaAngle=float(self.Items['deltaAngle'].GetValue())
        parent.step=int(self.Items['step of data'].GetValue())
        parent.pup.timer.interval=int(self.Items['interval'].GetValue())
        parent.quantity1=self.choice1.GetString(self.choice1.GetSelection())
        parent.quantity2=self.choice2.GetString(self.choice2.GetSelection())
        self.Destroy()

    def OnCANCEL(self,event):
        print '#OnCANCEL#'
        self.Destroy()
