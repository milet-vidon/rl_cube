"""NVRTC source for independent bounded pose problems and PCHIP speed checks."""
SOLVE = r"""
struct V {double v[3];};
struct M {double v[3][3];};
struct T {M R; V p;};
__device__ V add(V a,V b){for(int i=0;i<3;i++)a.v[i]+=b.v[i];return a;}
__device__ V mul(V a,double s){for(int i=0;i<3;i++)a.v[i]*=s;return a;}
__device__ double dot(V a,V b){return a.v[0]*b.v[0]+a.v[1]*b.v[1]+a.v[2]*b.v[2];}
__device__ V cross(V a,V b){return {{a.v[1]*b.v[2]-a.v[2]*b.v[1],a.v[2]*b.v[0]-a.v[0]*b.v[2],a.v[0]*b.v[1]-a.v[1]*b.v[0]}};}
__device__ M eye(){return {{{1,0,0},{0,1,0},{0,0,1}}};}
__device__ M tr(M a){M b;for(int i=0;i<3;i++)for(int j=0;j<3;j++)b.v[i][j]=a.v[j][i];return b;}
__device__ M mm(M a,M b){M c={};for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)c.v[i][j]+=a.v[i][k]*b.v[k][j];return c;}
__device__ V mv(M a,V b){V c={};for(int i=0;i<3;i++)for(int k=0;k<3;k++)c.v[i]+=a.v[i][k]*b.v[k];return c;}
__device__ T compose(T a,T b){return {mm(a.R,b.R),add(a.p,mv(a.R,b.p))};}
__device__ T inv(T a){M m=tr(a.R);return {m,mul(mv(m,a.p),-1.)};}
__device__ M expm(V v){
 double t=sqrt(dot(v,v)),s=t>1.e-8?sin(t)/t:1.,c=t>1.e-8?(1.-cos(t))/(t*t):.5;
 M a=eye(),k={{{0,-v.v[2],v.v[1]},{v.v[2],0,-v.v[0]},{-v.v[1],v.v[0],0}}},k2=mm(k,k);
 for(int i=0;i<3;i++)for(int j=0;j<3;j++)a.v[i][j]+=s*k.v[i][j]+c*k2.v[i][j];return a;
}
__device__ V logm(M m){
 V v={{(m.v[2][1]-m.v[1][2])*.5,(m.v[0][2]-m.v[2][0])*.5,(m.v[1][0]-m.v[0][1])*.5}};
 double s=sqrt(dot(v,v)),c=(m.v[0][0]+m.v[1][1]+m.v[2][2]-1.)*.5;
 if(c<-.99999999){int k=0;for(int i=1;i<3;i++)if(m.v[i][i]>m.v[k][k])k=i;V a={};a.v[k]=sqrt(fmax(0.,(m.v[k][k]+1.)*.5));
 for(int i=0;i<3;i++)if(i!=k)a.v[i]=(m.v[k][i]+m.v[i][k])/(4.*a.v[k]);return mul(a,atan2(s,c)*(dot(a,v)<0?-1.:1.));}
 return mul(v,s>1.e-12?atan2(s,c)/s:1.);
}
__device__ T fk(const double *chain,const double *q,V *orig,V *axes){
 T a={eye(),{}};for(int j=0;j<8;j++){
  const double *d=chain+13*j;T b;
  for(int i=0;i<3;i++){b.p.v[i]=d[9+i];for(int k=0;k<3;k++)b.R.v[i][k]=d[3*i+k];}
  a=compose(a,b);
  if(j!=7){double sign=d[12],s=sin(q[j]*sign),c=cos(q[j]*sign);M rz={{{c,-s,0},{s,c,0},{0,0,1}}};
   if(orig){orig[j]=a.p;for(int i=0;i<3;i++)axes[j].v[i]=a.R.v[i][2]*sign;}a.R=mm(a.R,rz);}
 }return a;
}
__device__ double system(const double *chain,const double *q,T goal,const double *ref,int n,
 double H[7][7],double g[7],double *e1,double *e2){
 V orig[7],axis[7];T a=fk(chain,q,H?orig:nullptr,axis);
 V p=add(goal.p,mul(a.p,-1.)),r=logm(mm(goal.R,tr(a.R)));double re=sqrt(dot(r,r)),wr=n==4?.1:4.;
 *e1=sqrt(dot(p,p));*e2=re;double cost=900*dot(p,p)+wr*wr*dot(r,r);
 double w[7]={.3,.3,.5,.8,4.,10.,10.},diag[7],b[7];V jp[7],jr[7];
 for(int j=0;j<n;j++){
  double w2=n==4?1.e-6:.000064*w[j],d=q[j]-ref[j]; cost+=w2*d*d;diag[j]=w2;b[j]=-w2*d;
  if(j==6){double wrist=q[j]-.6108652381980153;cost+=.0064*wrist*wrist;diag[j]+=.0064;b[j]-=.0064*wrist;}
  if(H){jp[j]=cross(axis[j],add(a.p,mul(orig[j],-1.)));
   // Inverse right Jacobian of log(Rgoal Rcurrent^T).
   double k=re<1.e-4?1./12.:(1.-.5*re/tan(.5*re))/(re*re);
   V c=cross(r,axis[j]);jr[j]=add(add(axis[j],mul(c,.5)),mul(cross(r,c),k));
   g[j]=900*dot(jp[j],p)+wr*wr*dot(jr[j],r)+b[j];
  }
 }
 if(H)for(int i=0;i<n;i++)for(int j=0;j<n;j++)H[i][j]=900*dot(jp[i],jp[j])+wr*wr*dot(jr[i],jr[j])+(i==j?diag[i]:0.);
 return cost;
}
__device__ bool solve(const double *chain,const double *lim,double *q,T goal,int n){
 double ref[7],e1,e2,lambda=1.e-5;
 for(int j=0;j<n;j++)ref[j]=q[j]=fmax(lim[2*j],fmin(lim[2*j+1],q[j]));
 for(int it=0;it<120;it++){
  double H[7][7],g[7];double cost=system(chain,q,goal,ref,n,H,g,&e1,&e2);
  bool freeze[7]={};for(int j=0;j<n;j++)freeze[j]=(q[j]<=lim[2*j]+1.e-10 && g[j]<0)||(q[j]>=lim[2*j+1]-1.e-10 && g[j]>0);
  double step[7]={};
  for(int active=0;active<=n;active++){
   double A[7][8]={};for(int i=0;i<n;i++){for(int j=0;j<n;j++)A[i][j]=freeze[i]||freeze[j]?(i==j):H[i][j]+(i==j?lambda:0.);
     A[i][n]=freeze[i]?0.:g[i];}
   for(int k=0;k<n;k++){double pivot=A[k][k];for(int i=k+1;i<n;i++){double s=A[i][k]/pivot;
     for(int j=k+1;j<=n;j++)A[i][j]-=s*A[k][j];}}
   for(int i=n-1;i>=0;i--){double t=A[i][n];for(int j=i+1;j<n;j++)t-=A[i][j]*step[j];step[i]=t/A[i][i];}
   bool addbound=false;for(int j=0;j<n;j++)if(!freeze[j]&&((q[j]<=lim[2*j]+1.e-10 && step[j]<0)||(q[j]>=lim[2*j+1]-1.e-10 && step[j]>0))){freeze[j]=true;addbound=true;}if(!addbound)break;
  }
  double inf=0.;for(int j=0;j<n;j++)inf=fmax(inf,fabs(step[j]));
  if(inf<2.e-9)break;
  double alpha=fmin(1.,.45/inf),qnew[7],newcost=cost;
  for(int ls=0;ls<10;ls++,alpha*=.5){
   for(int j=0;j<7;j++)qnew[j]=j<n?fmax(lim[2*j],fmin(lim[2*j+1],q[j]+alpha*step[j])):q[j];
   newcost=system(chain,qnew,goal,ref,n,nullptr,nullptr,&e1,&e2);
   if(newcost<cost)break;
  }
  if(newcost<cost){for(int j=0;j<n;j++)q[j]=qnew[j];
    if(cost-newcost<1.e-14*fmax(1.,cost))break;lambda=fmax(1.e-10,lambda*(alpha>.8?.2:1.));}
  else {lambda*=10;if(lambda>1.e7)break;}
 }
 system(chain,q,goal,ref,n,nullptr,nullptr,&e1,&e2);
 return e1<=.002 && e2<=(n==4?.4363323129985824:.0174532925199433);
}
// Each lane owns its history and branch decisions. The two arms communicate only
// at the rigid transfer of the final lift transform to the support palm.
extern "C" __global__ void reset_ik(const double *pose,const double *model,double *out,int *bad,int N){
 int lane=blockDim.x*blockIdx.x+threadIdx.x;if(lane>=N)return;
 const double *d=pose+lane*4,*chain=model,*limit=model+208,*home=limit+28,*params=home+14;
 double q[14];for(int i=0;i<14;i++)q[i]=home[i];
 T grasp[2],support,mainStart;M yaw=expm({{0,0,d[3]}});V xyz={{d[0],d[1],d[2]-params[27]}};
 for(int s=0;s<2;s++){
  M r;for(int i=0;i<3;i++)for(int j=0;j<3;j++)r.v[i][j]=params[s*12+i*3+j];
  V offset;for(int i=0;i<3;i++)offset.v[i]=params[s*12+9+i];grasp[s]={mm(yaw,r),add(xyz,mv(yaw,offset))};
 }
 support=grasp[1];int k=0;
 for(int ph=0;ph<7;ph++){
  int steps=ph<5?12:24;T start[2],target[2];V logdiff[2];
  for(int s=0;s<2;s++){start[s]=fk(chain+104*s,q+7*s,nullptr,nullptr);target[s]=grasp[s];}
  if(ph<5){double above[5]={.06,.06,.04,.015,0},width[5]={.23,.23,.23,.20,.20};
   target[0].p.v[2]+=above[ph];target[1].p=add(target[1].p,mv(yaw,{{0,.098-width[ph],0}}));
   target[1].p.v[2]+=above[ph]+.15;
  }else if(ph==5){mainStart=start[0];target[0]=mainStart;
   target[0].p=add(mainStart.p,mv(yaw,{{params[24],0,params[25]}}));}
  else {target[1]=compose(compose(start[0],inv(mainStart)),support);}
  for(int s=0;s<2;s++)logdiff[s]=logm(mm(target[s].R,tr(start[s].R)));
  for(int j=0;j<=steps;j++,k++){
   double u=double(j)/steps;
   if(j)for(int s=0;s<2;s++){
    if((ph==5&&s==1)||(ph==6&&s==0))continue;
    T goal={mm(expm(mul(logdiff[s],u)),start[s].R),add(mul(start[s].p,1.-u),mul(target[s].p,u))};
    if(!solve(chain+104*s,limit+14*s,q+7*s,goal,ph==5?4:7)){
      bad[lane]=1+1000*ph+100*s+j;return;
    }
   }
   for(int i=0;i<14;i++)out[(lane*115+k)*14+i]=q[i];
  }
 } bad[lane]=0;
}
"""
RETIME = r"""
extern "C" __global__ void retime(const double *p,const double *slope,const double *limits,
 const double *vel,const double *acc,long long *lengths,int *bad,int hz,int N) {
 int lane=blockIdx.x,ph=blockIdx.y,j=threadIdx.x;
 int off[7]={0,13,26,39,52,65,90};int m=(ph<5?12:24);
 double seconds[7]={2.8,2.4,2.,1.6,2.4,4.,2.4};
 __shared__ double ratio[64];__shared__ int len;
 double sec=seconds[ph];p+=(lane*115+off[ph])*56;slope+=(lane*115+off[ph])*56;
 for(int tries=0;tries<8;tries++){
  int n=(int)ceil(sec*hz);if(n<2)n=2;
  if(n>10000)break; // reject a path with an unusable or nonfinite duration
  double r=0.,prev=0.,dv=0.;
  if(j<56)for(int i=0;i<=n;i++){
   double z=double(i)/n,t=fmax(0.,fmin(double(m),z*z*z*(10.+z*(-15.+6.*z))*m));int k=(int)t;if(k>=m)k=m-1; double u=t-k;
   int a=k*56+j,b=a+56;
   double value=(1.-u)*(1.-u)*((1.+2.*u)*p[a]+u*slope[a])+u*u*((3.-2.*u)*p[b]+(u-1.)*slope[b]);
   if(!isfinite(value)||value<limits[j*2]-1.e-12||value>limits[j*2+1]+1.e-12)r=1.e9;
   double d=value-prev;if(i)r=fmax(r,fabs(d)*hz/vel[j]);
   if(i>1)r=fmax(r,sqrt(fabs(d-dv)*hz*hz/acc[j]));
   prev=value;dv=d;
  }
  ratio[j]=r;__syncthreads();
  for(int stride=32;stride;stride/=2){if(j<stride)ratio[j]=fmax(ratio[j],ratio[j+stride]);__syncthreads();}
  double mx=ratio[0];__syncthreads();
  if(mx<=1.){if(j==0)lengths[lane*9+ph]=n+1;return;}
  if(mx>1000.||!isfinite(mx))break;
  sec*=mx*1.03;
 }
 if(j==0){atomicCAS(bad+lane,0,8000+ph);lengths[lane*9+ph]=2;}
}
"""
