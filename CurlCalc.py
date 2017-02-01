#!/usr/bin/env python 

# Python Class to Calculate  Curl for Tensors w.r.t Coordinates in Real Space Given Unequally Spaced Data

import numpy as np
import math

from numpy import linalg as La
from scipy.interpolate import Rbf



class CurlCalc:
  """Initialize Coordinates, Reciprocal Lattice Vectors a*,b* & c*.
     Please Note: If the individual components of the R-Lattice Vectors are in row format, please convert them to Column format by taking
     the transpose.
  """

  def __init__(self,Coord,aStar,bStar,cStar):

    self.Coord=Coord  # Initialize array for spatial coordinates 
    self.aStar,self.bStar,self.cStar=aStar,bStar,cStar # Initialize arrays for Reciprocal Lattice Vectors


  def RealSpace(self):
   """ Given a set of reciprocal lattice vectors, this method converts them to real space vectors """    
   aS,bS,cS=self.aStar,self.bStar,self.cStar
   Real=np.zeros((len(aS),3,3))
   for i in range(len(aS)):
     Real[i,2,:]=np.cross(aS[i,:],bS[i,:])
     Real[i,0,:]=np.cross(bS[i,:],cS[i,:])
     Real[i,1,:]=np.cross(Real[i,2,:],Real[i,0,:])
   # Convert to unit vectors   
     Real[i,2,:]=Real[i,2,:]/(La.norm(Real[i,2,:]))
     Real[i,1,:]=Real[i,1,:]/(La.norm(Real[i,1,:]))
     Real[i,0,:]=Real[i,0,:]/(La.norm(Real[i,0,:]))
  
   return Real # Resulting Real Space Vectors


  def DefGrad(self,BIdeal,B):
   """ Method to calculate Elastic Deformation Gradient Given Real Space Vectors in the Undeformed and Deformed Condition"""
   F=np.zeros((len(B),3,3))
   for i in range(len(F)):
     F[i,:]=np.dot(B[i,:],La.inv(BIdeal))
   return F
    


  def Der(self,V,x): 
   """ Method to calculate derivative  w.r.t given set of coordinates: Uses Finite difference for unequally spaced datasets
    Source: http://websrv.cs.umt.edu/isis/index.php/Finite_differencing:_Introduction"""
   dx1=np.longdouble(x[1]-x[0])
   dx2=np.longdouble(x[2]-x[1])
   Diff=(-dx1**2*V[2]+(dx1+dx2)**2*V[1]-(dx1**2+2*dx1*dx2))/(dx1*dx2*(dx1+dx2))
    
   return Diff
   

  def DerLagrange(self,V,x):
    """ Method to calculate derivative  using 2nd Order Lagrange Polynomial  """
    
    dx1=np.longdouble(x[1]-x[2])  # (x(i-1)-x(i))
    dx2=np.longdouble(x[1]-x[3])  # (x(i-1)-x(i+1))
    dx3=np.longdouble(x[2]-x[3])  # (x(i)-x(i+1))
    
    Diff=(2*x[0]-x[2]-x[3])*V[1]/(dx1*dx2)+(2*x[0]-x[1]-x[3])*V[2]/(-dx1*dx3)+\
         (2*x[0]-x[1]-x[2])*V[3]/(dx2*dx3)
    
    
    return Diff    
  
  def PartialDeriv(self,X,fX,delX,delY,delZ,P):
    """ Function to Calculate Partial Derivatives Using Central Difference for Equal Intervals
      using a 5-point Stencil. http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/central-differences/ 
      X: Coordinate x,y,z
      fX: Values of Function at given coordinates
      delX,delY,delZ: Step Size for x,y & z
      P=P(x,y,z): Point at which the derivative is desired. """
    x,y,z=X[:,0],X[:,1],X[:,2]
    Z=Rbf(x,y,z,fX) # Use Radial basis function to interpolate function based on given data points
    # Calculate the Partial Derivative w.r.t x
    fxM2=Z(P[0]-2*delX,P[1],P[2]) # F(i-2)
    fxM1=Z(P[0]-delX,P[1],P[2])   # F(i-1)
    fxP1=Z(P[0]+delX,P[1],P[2])   # F(i+1)
    fxP2=Z(P[0]+2*delX,P[1],P[2]) # F(i+2)
    Diffx=(fxM2-8*fxM1+8*fxP1-fxP2)/(12.0*delX) # Derivative w.r.t x
  
    # Calculate the Partial Derivative w.r.t y
    fyM2=Z(P[0],P[1]-2*delY,P[2]) # F(i-2)
    fyM1=Z(P[0],P[1]-delY,P[2])   # F(i-1)
    fyP1=Z(P[0],P[1]+delY,P[2])   # F(i+1)
    fyP2=Z(P[0],P[1]+2*delY,P[2]) # F(i+2)
    Diffy=(fyM2-8*fyM1+8*fyP1-fyP2)/(12.0*delY) # Derivative w.r.t y
  
    # Calculate the Partial Derivative w.r.t z
    fzM2=Z(P[0],P[1],P[2]-2*delY) # F(i-2)
    fzM1=Z(P[0],P[1],P[2]-delY)   # F(i-1)
    fzP1=Z(P[0],P[1],P[2]+delY)   # F(i+1)
    fzP2=Z(P[0],P[1],P[2]+2*delY) # F(i+2)
    Diffz=(fzM2-8*fzM1+8*fzP1-fzP2)/(12.0*delZ) # Derivative w.r.t x
  
    return Diffx,Diffy,Diffz
     
     
    
  def OrMat(self,Euler):
    """ Method to Determine Orientation Matrix, given Euler angles in radians"""
    cp1,cph,cp2=np.cos(Euler[0]),np.cos(Euler[1]),np.cos(Euler[2])
    sp1,sph,sp2=np.sin(Euler[0]),np.sin(Euler[1]),np.sin(Euler[2])
    g11,g12,g13=(cp1*cp2-sp1*sp2*cph),(sp1*cp2+cp1*sp2*cph),(sp2*sph)
    g21,g22,g23=(-cp1*sp2-sp1*cp2*cph),(-sp1*sp2+cp1*cp2*cph),(cp2*sph)
    g31,g32,g33=(sp1*sph),(-cp1*sph),cph
    g=np.array([[g11,g12,g13],[g21,g22,g23],[g31,g32,g33]])
    return g
    
  def HexSymm(self,g):
     """ Method to Output result of Hexagonal symmetry operation of orientation matrix """
     # 12 Symmetry Operation Matrices for hcp systems: Adapted from Hagege et al. 1980
     S01=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
     S02=np.array([[0.5,0.87,0.],[-0.87,0.5,0.],[0.,0.,1.]])
     S03=np.array([[-0.5,0.87,0.],[-0.87,-0.5,0.],[0.,0.,1.]])
     S04=np.array([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]])
     S05=np.array([[-0.5,-0.87,0.],[0.87,-0.5,0.],[0.,0.,1.]])
     S06=np.array([[0.5,-0.87,0.],[0.87,0.5,0.],[0.,0.,1.]])
     S07=np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
     S08=np.array([[0.5,0.87,0.],[0.87,-0.5,0.],[0.,0.,-1.]])
     S09=np.array([[-0.5,0.87,0.],[0.87,0.5,0.],[0.,0.,-1.]])
     S10=np.array([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])
     S11=np.array([[-0.5,-0.87,0.],[-0.87,0.5,0.],[0.,0.,-1.]])
     S12=np.array([[0.5,-0.87,0.],[-0.87,-0.5,0.],[0.,0.,-1.]])
     GMat=np.zeros((12,3,3))
    
     GMat[0,:,:],GMat[1,:,:],GMat[2:,:]=np.dot(S01,g),np.dot(S02,g),np.dot(S03,g)     
     GMat[3,:,:],GMat[4,:,:],GMat[5,:,:]=np.dot(S04,g),np.dot(S05,g),np.dot(S06,g) 
     GMat[6,:,:],GMat[7,:,:],GMat[8,:,:]=np.dot(S07,g),np.dot(S08,g),np.dot(S09,g) 
     GMat[9,:,:],GMat[10,:,:],GMat[11,:,:]=np.dot(S10,g),np.dot(S11,g),np.dot(S12,g) 
    
     return GMat
    
  def MisOr(self,Euler1,Euler2):
    """ Method to determine Disorientation between two grains/points for Hexagonal Symmetry """
    gA=self.OrMat(Euler1)
    gB=self.OrMat(Euler2)
    g=np.dot(gB,La.inv(gA))
    GM=self.HexSymm(g)
    Ang=np.zeros((12,)) # Initialize Misorientation Array
   # Rounding to 2 Decimal places is done in order to ensure that values stay within [-1,1] interval!
    Ang[0]=math.acos(round((GM[0,0,0]+GM[0,1,1]+GM[0,2,2]-1.)*0.5,2))
    Ang[1]=math.acos(round((GM[1,0,0]+GM[1,1,1]+GM[1,2,2]-1.)*0.5,2))
    Ang[2]=math.acos(round((GM[2,0,0]+GM[2,1,1]+GM[2,2,2]-1.)*0.5,2))
    Ang[3]=math.acos(round((GM[3,0,0]+GM[3,1,1]+GM[3,2,2]-1.)*0.5,2))
    Ang[4]=math.acos(round((GM[4,0,0]+GM[4,1,1]+GM[4,2,2]-1.)*0.5,2))
    Ang[5]=math.acos(round((GM[5,0,0]+GM[5,1,1]+GM[5,2,2]-1.)*0.5,2))
    Ang[6]=math.acos(round((GM[6,0,0]+GM[6,1,1]+GM[6,2,2]-1.)*0.5,2))
    Ang[7]=math.acos(round((GM[7,0,0]+GM[7,1,1]+GM[7,2,2]-1.)*0.5,2))
    Ang[8]=math.acos(round((GM[8,0,0]+GM[8,1,1]+GM[8,2,2]-1.)*0.5,2))
    Ang[9]=math.acos(round((GM[9,0,0]+GM[9,1,1]+GM[9,2,2]-1.)*0.5,2))
    Ang[10]=math.acos(round((GM[10,0,0]+GM[10,1,1]+GM[10,2,2]-1.)*0.5,2))
    Ang[11]=math.acos(round((GM[11,0,0]+GM[11,1,1]+GM[11,2,2]-1.)*0.5,2))
    
    Ang=Ang*(180./np.pi)
    Ang=np.min(np.abs(Ang))
      
    return Ang
    
  def MisArray(self,Euler,EArray):
    M=[self.MisOr(Euler,EArray[ii]) for ii in range(len(EArray))]
    M=np.asarray(M)
    return M  
     
  def CurlTens(self,T,X,delX,delY,delZ): 
    """ Method to Compute Curl of a 2nd RankTensor with respect to a Real Space Vector
        Return a single set of 9 components for each point evaluated"""
  
    Curl =np.empty((len(T),3,3))  #Initialize the Array to Store the Calculated Curl
    
    # Extract the 9 Components of the Tensor
    T11,T22,T33=T[:,0,0],T[:,1,1],T[:,2,2] 
    T12,T13,T23=T[:,0,1],T[:,0,2],T[:,1,2] 
    T21,T31,T32=T[:,1,0],T[:,2,0],T[:,2,1] 
  
    #Calculate Individual Elements of the Curl Tensor
    
    P=np.array([X[0,0],X[0,1],X[0,2]])
    #D2T31,D3T21,D2T32=self.DerLagrange(T31,a2),self.DerLagrange(T21,a3),self.DerLagrange(T32,a2)
    #D3T22,D2T33,D3T23=self.DerLagrange(T22,a3),self.DerLagrange(T33,a2),self.DerLagrange(T23,a3)
    #D3T11,D1T31,D3T12=self.DerLagrange(T11,a3),self.DerLagrange(T31,a1),self.DerLagrange(T12,a3)
    #D1T32,D3T12,D1T33=self.DerLagrange(T32,a1),self.DerLagrange(T12,a3),self.DerLagrange(T33,a1)
    #D1T21,D2T11,D1T22=self.DerLagrange(T21,a1),self.DerLagrange(T11,a2),self.DerLagrange(T22,a1)
    #D2T12,D1T23,D2T13=self.DerLagrange(T12,a2),self.DerLagrange(T23,a1),self.DerLagrange(T13,a2)
    DxT31,DyT31,DzT31=self.PartialDeriv(X,T31,delX,delY,delZ,P)
    DxT21,DyT21,DzT21=self.PartialDeriv(X,T21,delX,delY,delZ,P)
    DxT32,DyT32,DzT32=self.PartialDeriv(X,T32,delX,delY,delZ,P)
    DxT22,DyT22,DzT22=self.PartialDeriv(X,T22,delX,delY,delZ,P)
    DxT33,DyT33,DzT33=self.PartialDeriv(X,T33,delX,delY,delZ,P)
    DxT23,DyT23,DzT23=self.PartialDeriv(X,T23,delX,delY,delZ,P)
    DxT11,DyT11,DzT11=self.PartialDeriv(X,T11,delX,delY,delZ,P)
    DxT12,DyT12,DzT12=self.PartialDeriv(X,T12,delX,delY,delZ,P)
    DxT13,DyT13,DzT13=self.PartialDeriv(X,T13,delX,delY,delZ,P)
    
    
    
    C11,C12,C13=(DyT31-DzT21),(DyT32-DzT22),(DyT33-DzT23)
    C21,C22,C23=(DzT11-DxT31),(DzT12-DxT32),(DzT12-DxT33)
    C31,C32,C33=(DxT21-DyT11),(DxT22-DyT12),(DxT23-DyT13)
    
    Curl=np.array([[C11,C12,C13],[C21,C22,C23],[C31,C32,C33]])
     
     
     
    return Curl
  


