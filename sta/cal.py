#!/usr/bin/env python
from scipy.spatial.distance import cdist as dis
import math
import numpy as np
import numpy.random as nr
from scipy import spatial
from scipy.stats import t
from scipy import linalg
import sys
import csv
import wx
import time

class calculate(object):

    figurepanel = None
    parent = None
    neighborRadius=None
    Rmax = 1
    Fmax = 1

    @staticmethod
    def selectdata(num):
        totaldata = len(calculate.parent.data)
        if totaldata <= num:
            selectedindex = np.arange(0, totaldata)
        else:
            selectedindex = np.array(np.linspace(0,totaldata-1,num),dtype=int)
        return selectedindex

    @staticmethod
    def readdata(step=1):
        try:
            with open(calculate.parent.inputpath, 'r') as f:
                reader = csv.reader(f)
                fr=1
                buff = []
                for ind,row in enumerate(reader):
                    NotReadable = False
                    for index, xd in enumerate(row):
                        if ind==0:
                            calculate.Fmax = float(xd)
                            NotReadable = True
                            break
                        try:
                            row[index] = float(xd)
                        except:
                            NotReadable = True
                            break
                    if NotReadable:
                        print 'useless data++'
                        continue
                    if row[4] == fr:
                        buff.append(row[:4])
                    elif row[4] > fr:
                        fr+=step
                        yield np.array(buff)
                        if row[4]==fr-step+1:
                            buff=[row[:4]]
                        else:
                            buff=[]
                    else:
                        pass
                f.close()

        except IOError:
            info = sys.exc_info()
            print info[0], ':', info[1]
            wx.MessageBox('Could not open this file.', 'alert', wx.OK)
            calculate.parent.statusbar.SetStatusText('Ready.Data field: ;Result field:' + calculate.parent.inputpath)

    @staticmethod
    def init(panel,parent):
        calculate.figurepanel = panel
        calculate.parent = parent

    @staticmethod
    def cal_density():#a function to calculate the density of whole area
        print '#cal_density#'
        data=calculate.readdata(calculate.parent.step)
        den = []
        for idx,the in enumerate(data):#for each frame
            points=the[:,0:2]#positions of points
            qhull = spatial.Delaunay(points)
            area = sum([abs(np.cross(t[0],t[1]))*0.5 for t in points[qhull.simplices]])
            #print area
            num = len(points)
            den.append(float(num)*1000/area)
            """
            cKDpoints=spatial.cKDTree(points)
            center = np.mean(points, 0)
            commonR = np.mean(np.array(spatial.distance.cdist([center], points)[0]))#average distance between the center and points
            deltaR = commonR *calculate.parent.deltaRadius#step of selected r from the center
            dR = deltaR * 2.#width of the layer at r to count number of points between r and r+dR
            R_max = 3*commonR + deltaR / 2.0
            R = np.arange(deltaR,R_max , deltaR)
            numbersPlus=cKDpoints.count_neighbors(spatial.cKDTree([center]), R+dR)#number of points whose distance frome the center is less than r+dR, r is in R
            numbers=cKDpoints.count_neighbors(spatial.cKDTree([center]), R)#number of points whose distance frome the center is less than r, r is in R
            density=(numbersPlus-numbers)/ (math.pi * dR * (2 * R + dR))
            yield [[R.tolist(),density.tolist()]]
            """
        return [range(len(den)),den]

    @staticmethod
    def cal_order():#the order parameter for each frame which is the same as the polarization
        print '#cal_order#'
        data=calculate.readdata(calculate.parent.step)
        order_para = []
	for frame in data:
	    frame = np.array(frame)
            velocities = frame[0:,2:4]
            speeds = np.sqrt(np.sum(velocities**2,1))
            order = np.linalg.norm(np.sum(velocities,0))/np.sum(speeds)
            order_para.append(order)
        length=len(order_para)
        return [range(length),order_para]

    @staticmethod#the polarization for each frame which is the same as the order parameter
    def cal_polarization():
        print '#cal_polarization#'
        data=calculate.cal_PolarizationOrOrder()
        length=len(data)
        return [range(length), data]

    @staticmethod
    def cal_PolarizationOrOrder():#calculate the polarization and the order parameter which are the same
        data = calculate.readdata(calculate.parent.step)
        polarization_order=[]
        for frame in data:
            frame = np.array(frame)
            velocities = frame[0:, 2:4]
            speeds=np.sqrt(np.sum( velocities ** 2, 1))#magnitude of velocity of each point
            velocities[0:,0]/=speeds#cos(alpha) alpha is the angle between the velocity and axis X
            velocities[0:,1]/=speeds#sin(alpha)
            angles = velocities
            #print np.sum(angles**2,1)
            a = math.sqrt(np.sum(np.mean(angles,0)**2))
            polarization_order.append(a)
        return polarization_order

    @staticmethod
    def cal_pairDist():#the pairwise distribution p(r),a generator
        print '#cal_pairDist#'

        # define to calculate max distance between neighbors
        def maxdis(pt):
            delau = spatial.Delaunay(pt)
            convex_b = delau.convex_hull
            convex_b.shape=-1
            convex_b = np.array({}.fromkeys(convex_b).keys())
            dis = np.array([np.linalg.norm(pt[convex_b]-pt[p],axis=1) for p in convex_b])
            return np.max(dis)

        data=calculate.readdata(calculate.parent.step)
        for idx,the in enumerate(data):#for each frame
            points=the[:,0:2]
            cKDpoints=spatial.cKDTree(points)
            maxR = maxdis(points)
            deltaR = maxR / float(calculate.parent.NumOfRadius)
            R_max=maxR+deltaR/2.
            R = np.arange(0, R_max, deltaR)
            """
            if idx==0:
            	calculate.Rmax = maxdis(points)
            dr = calculate.Rmax/float(calculate.parent.NumOfRadius)
            R = np.arange(0,calculate.Rmax+dr,dr)
            """
            pairsnumber=cKDpoints.count_neighbors(cKDpoints,R)#number of points whose distance frome the center is less than r, r is in R
            pairsnumber = pairsnumber[1:] - pairsnumber[:-1]#number of points whose distance frome the center is between r and r+deltaR, r is in R
            pairsdensity = pairsnumber/(np.pi*(-(R[1:]-deltaR)**2+R[1:]**2))
            #print [[R[:-1].tolist(),pairsnumber.tolist()]]
            yield [R[1:].tolist()-deltaR/2.0,pairsdensity.tolist(),calculate.Fmax/calculate.parent.step]

    @staticmethod
    def cal_densityAngular():#the density at different directions , a generator
        print '#cal_densityAngular#'
        data=calculate.readdata(calculate.parent.step)
        for idx,the in enumerate(data):
            points=the[:,0:2]
            cKDpoints=spatial.cKDTree(points)
            center = np.mean(points, 0)
            commonR = np.mean(np.array(spatial.distance.cdist([center], points)[0]))
            directions = np.arctan2(the[:,3],the[:,2])
            r = commonR/2.0
            samples=cKDpoints.query_ball_point(center,r)#selecte the points close to the center
            neighbors=spatial.cKDTree(points[samples]).query_ball_tree(cKDpoints,r)#find the neighbors around each selected point
            for i in range(len(neighbors)):
                neighbors[i]=list(set(neighbors[i])-set([samples[i]]))#the neighbors around a point should not include itself
            deltaAngle=math.pi*calculate.parent.deltaAngle
            angleArray=np.arange(-math.pi, math.pi, deltaAngle)
            Distributions=[]
            for i,centerindex in enumerate(samples):#for each selected point
                point = points[centerindex]
                dirt = directions[centerindex]
                neighborpoints=points[neighbors[i]]
                deltaX = neighborpoints-point
                relativedirts = np.arctan2(deltaX[:,1],deltaX[:,0])-dirt#angle between the arrow from the selected point to its neighbor and the velocity of the selected point
                for i,a in enumerate(relativedirts):#make the angle between -math.pi and math.pi
                    if a>math.pi:
                        relativedirts[i]-=math.pi*2
                    if a<=-1*math.pi:
                        relativedirts[i]+=math.pi*2
                OneDistribution=[]
                for angle in angleArray:
                    OneDistribution.append(len(np.where(np.logical_and(relativedirts>=angle,relativedirts<angle+deltaAngle))[0])/(0.5*deltaAngle*r*r))#density between angle and angle+deltaAngle
                Distributions.append(OneDistribution)
            AverageDistribution=np.mean(np.array(Distributions),0)
            yield [[angleArray.tolist(),AverageDistribution.tolist()]]

    @staticmethod
    def cal_numberAngular():#number of points with different direction of velocity,a generator
        print '#cal_numberAngular#'
        data=calculate.readdata(calculate.parent.step)
        for idx, the in enumerate(data):
            directions = np.arctan2(the[:, 3], the[:, 2])#direction of velocity
            deltaAngle = math.pi * calculate.parent.deltaAngle
            angleArray = np.arange(-math.pi, math.pi, deltaAngle)
            Distribution = [len(np.where(np.logical_and(  directions>angle ,directions<angle+deltaAngle))[0]) for angle in angleArray]
            yield [[angleArray.tolist(),Distribution]]

    @staticmethod
    def cal_inteCondition():#Integrated conditional density at different radius,a generator
        print '#cal_inteCondition#'
        data=calculate.readdata(calculate.parent.step)
        for idx,the in enumerate(data):
            ind_s = np.random.randint(0,len(the),np.int(len(the)*0.5))
            points=the[ind_s,0:2]
            cKDpoints=spatial.cKDTree(points)
            center = np.mean(points, 0)
            commonR = np.mean(np.array(spatial.distance.cdist([center], points)[0]))
            deltaR = commonR /calculate.parent.NumOfRadius
            R_max=commonR + deltaR / 2.0
            R = np.arange(deltaR, R_max, deltaR)
            #sampleIndex = cKDpoints.query_ball_point(center, commonR)#selecte the points close to the center
            selected_num=[]
            selected_num=cKDpoints.count_neighbors(cKDpoints,R)/float(len(points))#number of points whose distance from the selected point is less than r in R
            #print selected_num,R
            intedensity=(np.array(selected_num)*1000 / (math.pi * R*R))#calculate the average value of all the selected points
            yield [R.tolist(),intedensity.tolist(),calculate.Fmax/calculate.parent.step]

    @staticmethod
    def cal_averageV_value():#magnitude of average velocity of all the points
        print '#cal_averageV#'
        data = calculate.readdata(calculate.parent.step)
        averSpeed=[]
        for frame in data:
            frame = np.array(frame)
            velocities = frame[0:,2:4]
            V=np.mean(velocities,0)
            averSpeed.append(math.sqrt(np.sum(V**2)))
        return range(len(averSpeed)),averSpeed

    @staticmethod
    def cal_averageV_angle():#direction of average velocity of all the points
        print '#cal_averageV#'
        data = calculate.readdata(calculate.parent.step)
        averAngle = []
        for frame in data:
            frame = np.array(frame)
            velocities = frame[0:, 2:4]
            V = np.mean(velocities, 0)
            averAngle.append(math.atan2(V[1], V[0]))
        return range(len(averAngle)),averAngle

    @staticmethod
    def cal_neardist():#average value of the nearest distance from others of each point
        print '#cal_neardist#'
        data = calculate.readdata(calculate.parent.step)
        nearDist=[]
        for frame in data:
            frame=np.array(frame)
            points = frame[0:,0:2]
            cKDpoints=spatial.cKDTree(points)
            dist=cKDpoints.query(points,2)[0]#calculate the nearest 2 neighbors of each points(the first is itself)
            nearDist.append(np.mean(dist[:,1]))
        framesnumber = len(nearDist)
        Fr = range(framesnumber)
        return [Fr,nearDist]

    @staticmethod
    def cal_correlation():#correlation of two quantities
        a=np.array(eval('calculate.'+calculate.parent.pdown.ListContent[calculate.parent.quantity1])()[1])
        b=np.array(eval('calculate.'+calculate.parent.pdown.ListContent[calculate.parent.quantity2])()[1])#a and b are the two quantities we are going to calculate
        aver_a=np.mean(a,0)
        aver_b=np.mean(b,0)
        length=min(len(a),len(b))
        result = np.correlate(a-aver_a,b-aver_b,'full')
        Fr = np.array(range(len(result)))+1
        result = result/(length-abs(Fr-length))
        #length=len(a)
        #Fr=range(length-1)
        #result=[]
        #for t in Fr:
        #    result.append(np.mean((b[t:]-aver_b)*(a[:length-t]-aver_a),0))
        return [Fr-len(result)/2,result.tolist()]
###################################################################################################################################

    @staticmethod
    def cal_vor():#vorticity at position of each point
        print '#cal_vor#'
        data=calculate.readdata(calculate.parent.step)
        for the in data:
                points=the[:,0:2]
                cKDpoints=spatial.cKDTree(points)
                center=np.mean(points,0)
                commonR = np.mean(np.array(spatial.distance.cdist([center],points)[0]))
                velocities = the[0:, 2:4]
                r = commonR *calculate.parent.neighborRadius
                vor_array = []
                all_neighbors=cKDpoints.query_ball_tree(cKDpoints,r)#neighbors of each point
                for index,p in enumerate(points):
                    neighbors = all_neighbors[index]#neighbors of this point
                    neardeltapoints = points[neighbors] - p#delta X
                    NeardeltaV = velocities[neighbors] - velocities[index]#delta V
                    absdeltaV=np.sqrt(np.sum( np.array(NeardeltaV) **2,1))#magnitude of delta V
                    vXx=neardeltapoints[:,0]*NeardeltaV[:,1]-neardeltapoints[:,1]*NeardeltaV[:,0]#multiplicaiton cross
                    vXunitx=[]
                    for i in range(len(neighbors)):
                        if absdeltaV[i]==0:
                            vXunitx.append(0)
                        else:
                            vXunitx.append(vXx[i]/absdeltaV[i])
                    vor_array.append(np.mean(np.array(vXunitx),0))
                vor_array=np.array(vor_array).reshape(-1,1)
                yield np.hstack((points,vor_array)).tolist()
               
    '''
    @staticmethod
    def cal_vor2():
        print '#cal_vor2#'
        if type == 'plot':
            sele=calculate.selectdata(1)
            data=calculate.parent.data[sele]
            idxs=np.arange(len(calculate.parent.data))[sele]
        elif type == 'save':
            data=calculate.parent.data
            idxs=range(len(data))
        else:
            return
        result=[]
        for idx,the in enumerate(data):
            points=the[:,0:2]
            cKDpoints=spatial.cKDTree(points)
            center = np.mean(points, 0)
            commonR = np.mean(np.array(spatial.distance.cdist([center], points)[0]))
            velocities = the[0:, 2:4]
            r = commonR * calculate.parent.neighborRadius
            vor_array = []
            for x_i in np.linspace(0,2000,100):
                line=[]
                for y_i in np.linspace(0,2500,100):
                    p = [x_i,y_i]
                    neighbors = cKDpoints.query_ball_point(p, r)#0000
                    if len(neighbors)==0:
                        line.append(0)
                        continue
                    neardeltapoints = points[neighbors] - p
                    neardeltaV = velocities[neighbors]
                    absdeltaV = np.sqrt(np.sum(neardeltaV ** 2, 1))
                    vXx = neardeltapoints[:, 0] * neardeltaV[:, 1] - neardeltapoints[:, 1] * neardeltaV[:, 0]
                    vXunitx = []
                    for i in range(len(neighbors)):
                        if absdeltaV[i] == 0:
                            vXunitx.append(0)
                        else:
                            vXunitx.append(vXx[i] / absdeltaV[i])
                    line.append(np.mean(np.array(vXunitx)))
                vor_array .append(line)
            result.append(vor_array)

        Information = {'dataType': 'Picture', 'dataMode': '[[value...],...]', 'Title': 'Vorticity', 'xlabel': 'x',
                       'ylabel': 'y', 'legend': False}
        return {'Information': Information, 'data': result}
    '''
    @staticmethod
    def cal_dv():#divergence at position of each point
        print '#cal_dv#'
        data=calculate.readdata(calculate.parent.step)
        for idx, the in enumerate(data):
            points = the[:, 0:2]
            cKDpoints = spatial.cKDTree(points)
            center = np.mean(points, 0)
            commonR = np.mean(np.array(spatial.distance.cdist([center], points)[0]))
            velocities = the[0:, 2:4]
            r = commonR *calculate.parent.neighborRadius
            Dv_array=[]
            all_neighbors=cKDpoints.query_ball_tree(cKDpoints,r)#neighbors of each point
            for index,p in enumerate(points):
                neighbors = all_neighbors[index]#neighbors  of  this  point
                neardeltapoints=points[neighbors]-p#delta X
                neardeltaV=velocities[neighbors]-velocities[index]#delta V
                Dv_array.append(np.mean(np.sum(neardeltapoints*neardeltaV,1)))
            Dv_array = np.array(Dv_array).reshape(-1, 1)
            yield  np.hstack((points, Dv_array)).tolist()


    @staticmethod
    def cal_localAverV():#local average Velocity around each point
        print '#cal_localAverV#'
        data=calculate.readdata(calculate.parent.step)
        for idx, the in enumerate(data):
            points = the[:, 0:2]
            cKDpoints = spatial.cKDTree(points)
            center = np.mean(points, 0)
            commonR = np.mean(np.array(spatial.distance.cdist([center], points)[0]))
            velocities = the[0:, 2:4]
            r = commonR*calculate.parent.neighborRadius
            average_NV=[]
            Num=range(len(points))
            all_neighbors=cKDpoints.query_ball_tree(cKDpoints,r)#neigbors of each point
            for index in Num:
                neighborsV=velocities[all_neighbors[index]]
                average_NV.append(np.mean(np.array(neighborsV),0))
            yield  [np.hstack((points, np.array(average_NV))).tolist()]

    @staticmethod
    def cal_Gv(sv=False):
        print '#cal_Gv#'
        data=calculate.cal_localAverV()
        for idx, the in enumerate(data):
            the=np.array(the[0])
            points = the[:, 0:2]
            center = np.mean(points, 0)
            commonR = np.mean(np.array(spatial.distance.cdist([center], points)[0]))
            r = commonR * calculate.parent.neighborRadius
            localV=the[:,2:4]
            cKDpoints=spatial.cKDTree(points)
            nei=cKDpoints.query_ball_tree(cKDpoints,r)#neighbors of each point
            dis,idxs=cKDpoints.query(points,8)
            gv=[]
            """
            for pointIdx,nearIdx in enumerate(idxs):
                if len(nearIdx)<2:
                    gv.append([[0,0],[0,0]])
                else:
                    gv.append(linalg.lstsq(points[nearIdx]-points[pointIdx],localV[nearIdx]-localV[pointIdx])[0])#least square solution
                    #print localV[nearIdx]-localV[pointIdx]
                    #print linalg.lstsq(points[nearIdx]-points[pointIdx],localV[nearIdx]-localV[pointIdx])[1]
            """
            for pointIdx,nearIdx in enumerate(idxs):
                nearIdx = nearIdx.tolist()
                nearIdx.remove(pointIdx)
                #dv = localV[nearIdx]-localV[pointIdx]
                v = np.linalg.norm(localV[nearIdx],axis=1)-np.linalg.norm(localV[pointIdx])
                dl = points[pointIdx]-points[nearIdx]
                length = np.hypot(dl[:,0],dl[:,1])
                grads = v/length
                theta = np.arctan2(dl[:,1],dl[:,0])
                gradx = np.sum(grads*np.cos(theta))
                grady = np.sum(grads*np.sin(theta))
                gv.append([gradx.tolist(),grady.tolist()])
                #print [gradx,grady]
            gv=np.array(gv)
            if not sv:
                result = np.hstack((points,gv))
            else:
                result = np.hstack((points,gv,localV))
            #gv.shape=-1,4
            #resultx=np.hstack((points,gv[:,0].reshape(-1,1),gv[:,2].reshape(-1,1),localV[:,0].reshape(-1,1))).tolist()
            #resulty=np.hstack((points,gv[:,1].reshape(-1,1),gv[:,3].reshape(-1,1),localV[:,1].reshape(-1,1))).tolist()
            yield [result.tolist()]

    @staticmethod
    def cal_Sv():
        print '#cal_Sv'
        data=calculate.cal_Gv(True)
        #for xy in data:
        for gv in data:    # for every frame
            """
            x=np.array(xy[0])
            y=np.array(xy[1])
            points = x[:,0:2]
            gv_x=x[:,2:4]
            gv_y=y[:,2:4]
            ly=x[:,4]*(-1.)
            lx=y[:,4]
            l=np.hstack(((lx/np.sqrt(lx**2+ly**2)).reshape(-1,1) , (ly/np.sqrt(lx**2+ly**2)).reshape(-1,1)))
            """
            gv = np.array(gv[0])
            pos = gv[:,0:2]
            localv = gv[:,4:6]
            localv_perpen = np.array([-localv[:,1],localv[:,0]]).T
            GV = gv[:,2:4]
            SV = np.sum(GV*localv_perpen,1)/np.linalg.norm(localv,axis=1)
            re = np.array([pos[:,0],pos[:,1],SV]).T
            yield re
            #yield  [np.hstack((points, np.sum(l*gv_x,1).reshape(-1,1) , np.sum(l*gv_y,1).reshape(-1,1) )).tolist()]

    @staticmethod
    def cal_cluster():
        print '#cal_cluster#'
        '''
                def Sil(result):  # the first function to get k. not useful
                    distances = dis(points, points)
                    si = []
                    for idxs_i, idxs in enumerate(result):
                        for i in idxs:
                            if len(idxs) == 1:
                                a = 0
                            else:
                                a = np.mean(distances[i][idxs]) / (len(idxs) - 1) * len(idxs)
                            b = np.min(
                                np.array([np.mean(distances[i][idx]) for idx_i, idx in enumerate(result) if idx_i != idxs_i]))
                            si.append((b - a) / max(a, b))
                    return np.mean(np.array(si))

                def DB(centers,result):  # the second function to get k. accurate but not easy to read k from the graph. and may be bad sometimes
                    distances = dis(centers, centers)
                    N = range(len(centers))
                    Dw = [np.mean(dis([centers[i]], points[result[i]])[0], 0) for i in N]
                    return np.mean(np.array([np.max(np.array([(Dw[i] + Dw[j]) / distances[i][j] for j in N if j != i]), 0) for i in N]))
        '''
        def F(center_ofall, centers,result):  # the third function to get k. easy to find k(min). but sometimes may be not accurate.
            a = np.sum(dis([center_ofall], centers)[0])
            b = np.sum(np.array([np.sum(dis([centers[i]], points[id])[0]) for i, id in enumerate(result)]))
            return (a + b)

        def k_means(points, k, cluster_centers=None):
            Num = len(points)
            if k >= Num - 1:
                k = k / 2
            a = np.arange(Num)
            if Num < 2:
                print 'error'
            if cluster_centers == None:
                cluster_centers = np.empty((k, 2))
                cluster_centers[0] = points[nr.choice(a)]
                for i in range(k):
                    if i == 0:
                        dist = [dis([points[index]], [cluster_centers[0]])[0][0] for index in a]
                    else:
                        cluster_centers[i] = points[ano_index]
                        dist = np.min(np.vstack((dis([cluster_centers[i]], points)[0], dist)), 0)
                    ano_index = nr.choice(a, p=dist / np.sum(dist))

            first = True
            while True:
                kd = spatial.cKDTree(cluster_centers)
                idx = kd.query(points, 1)[1]
                result = []
                for i in range(len(cluster_centers)):
                    result.append([])
                for i, id in enumerate(idx):
                    result[id].append(i)
                result = np.array(result)
                notempty = [i for i in range(len(result)) if len(result[i]) != 0]  # remove the empty clusters
                result = result[notempty]
                ano_centers = np.array([np.mean(points[id], 0) for id in result])
                this_d = np.mean(np.sqrt(np.sum((ano_centers - cluster_centers) ** 2, 1)))
                if not first:
                    if dmean * 0.001 >= this_d:  # must be >= not > ,because sometimes they are both 0.0
                        return cluster_centers, result
                dmean = this_d
                first = False
                cluster_centers = ano_centers

        def change(centers, idxs, points):  # merge
            kd = spatial.cKDTree(centers)
            dist, idx = kd.query(centers, 2)
            a, b = idx[np.argmin(dist[:, 1])]
            centers[a] = np.mean(points[np.concatenate((np.array(idxs[b]), np.array(idxs[a])))], 0)
            centers[b] = np.array(centers[-1])
            centers = centers[:-1]
            return k_means(points, 0, centers)

        data=calculate.readdata()
        for the in data:
            points=the[:,0:2]
            func = []  # function to get k
            k = []  # array of k
            centers_total = []  # save all the centers in the process
            centers, result = k_means(points, int(np.sqrt(len(points)) * 9))
            notempty = [i for i in range(len(result)) if len(result[i]) != 0]  # remove the empty cluster
            centers = (centers)[notempty]
            result = result[notempty]
            center_ofall = np.mean(points, 0)
            while len(centers) > 2:
                centers_total.append(centers)
                centers, result = change(centers, result, points)
                k.append(len(centers))
                func.append(F(center_ofall, centers, result))
                # func.append(DB(centers,result))
                # func.append(Sil(result))
            func = func[int(len(func) / 2.5):]  # reduce the range
            k = k[int(len(k) / 2.5):]
            func2 = np.array([func[i] + (-(func[i + 1] - func[i]) / (k[i + 1] - k[i]) + (func[i - 1] - func[i]) / (k[i - 1] - k[i])) / 4.0 for i in
                              range(len(func)) if i > 0 and i < len(func) - 1])  # smoothing
            k2 = np.array(k[1:-1])
            f = np.array(func2)
            kk = k2
            '''
            xielvC=(np.max(f)-np.min(f))/(kk[0]-kk[-1])
            for index1 in range(len(f)-1):
                if (f[index1 +1]- f[index1] )/(kk[index1+1]-kk[index1])>xielvC/2:
                    q += 1
                else:
                    q= 0
                if q==2:
                    break
            print index1

            b=np.array(f[index1:])
            a=np.argmin(f)
            print 'a',kk[index1:][a]
            centers_need=[centers_total[i] for i in range(len(centers_total)) if len(centers_total[i])==kk[index1:][a]][0]
            kd = spatial.cKDTree(centers_need)
            idx = kd.query(points, 1)[1]
            result = []
            for i in range(len(centers_need)):
                result.append([])
            for i, id in enumerate(idx):
                result[id].append(i)
            color = []
            x=[]
            y=[]
            for i, ps in enumerate(result):
                x += points[ps][:, 0].tolist()
                y += points[ps][:, 1].tolist()
                for il in range(len(ps)):
                    color.append(i)
            plt.figure(5)
            plt.scatter(x,y,c=color)
            plt.colorbar()
            plt.scatter(centers_need[:,0],centers_need[:,1],lw=4)
            plt.show()
            '''
            a = np.argmin(f)
            centers_need = [centers_total[i] for i in range(len(centers_total)) if len(centers_total[i]) == kk[a]][0]#selecte the data for best k
            kd = spatial.cKDTree(centers_need)
            idx = kd.query(points, 1)[1]#classify again according to the centers we get
            result = []
            for i in range(len(centers_need)):
                result.append([])
            for i, id in enumerate(idx):
                result[id].append(i)
            color = []
            x = []
            y = []
            for i, ps in enumerate(result):
                x += points[ps][:, 0].tolist()
                y += points[ps][:, 1].tolist()
                for il in range(len(ps)):
                    color.append(i)
            print len(x),len(y),len(color)
            yield np.hstack((np.array(x).reshape(-1,1),np.array(y).reshape(-1,1),np.array(color).reshape(-1,1))).tolist()

    @staticmethod
    def draw_sheep():
        print '#draw_sheep#'
        for a in calculate.readdata(calculate.parent.step):
            yield [a.tolist()]

    @staticmethod
    def error(array,axis=None,errorType='se'):#useless till now
        if errorType == 'sd':
            return np.std(array)
        elif errorType == 'se':
            return np.std(array) / np.sqrt(len(array))
        elif errorType == 'ci':
            return t.ppf(1 - (1-0.95)/ 2., len(array)-1) * np.std(array) * np.sqrt(1. + 1. / len(array))
