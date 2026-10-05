# Linearisation: margins and the sedan's modes, task 3.7

- **Date:** 5 October 2026, cloud session (Linux container, GCC 13.3; times
  from this container, not comparable with the laptop's).
- **Produced by:** the two programs below, compiled against the library of
  this task from the repository root:

  ```sh
  g++ -std=c++20 -O2 -Iinclude -Ibuild-linux/_deps/eigen-src lin.cpp \
      build-linux/src/libmultibody_core.a -o lin && ./lin
  g++ -std=c++20 -O1 -Iinclude -Itests -Ibuild-linux/_deps/eigen-src lincases.cpp \
      build-linux/src/libmultibody_core.a -o lincases && ./lincases
  ```

## The quarter car and the sedan (`lin`)

```
statics: converged in 1 Newton iterations and 0 relaxation steps; |a| 20.19 -> 1.828e-09, largest in v[0], the prismatic joint of body 1 (wheel)
history: 20.19 1.828e-09
degrees of freedom 2: 0 without stiffness, 0 unstable
linearisation: 2 degrees of freedom; operating point |a| 1.828e-09, |v| 0
mode 1: 1.382 Hz natural, 1.32 Hz damped, damping ratio 0.2941, largest in v[1], the prismatic joint of body 2 (chassis)
mode 2: 11.6 Hz natural, 11.19 Hz damped, damping ratio 0.2635, largest in v[0], the prismatic joint of body 1 (wheel)
undamped 1: 1.356 Hz, largest in v[1], the prismatic joint of body 2 (chassis)
undamped 2: 11.81 Hz, largest in v[0], the prismatic joint of body 1 (wheel)
analytic undamped 1.356369445435 11.811152065254 Hz
lin undamped 1.356369445435 Hz
lin undamped 11.811152065254 Hz
bounce shape ratio 0.092125651135 analytic 0.092125651135 err 2.37e-14
ref eig -19.196874756153 +70.285720455286 i
ref eig -19.196874756153 -70.285720455286 i
ref eig -2.553125243847 +8.296441947943 i
ref eig -2.553125243847 -8.296441947943 i
lin eig -2.553125243823 +8.296441947948 i  shape (0.078408+0.054296i, 1.000000+0.000000i)
lin eig -19.196874755992 +70.285720455356 i  shape (1.000000+0.000000i, -0.030224-0.076605i)
K=
220000 -20000
-20000  20000
C=
 1499.99999999 -1499.99999999
-1499.99999999  1499.99999999
M=
 40   0
  0 250
N=
1 0
0 1
B=
    0     0
    0     0
0.025     0
    0 0.004
linearisation: 10 degrees of freedom; operating point |a| 1.815e-08, |v| 0
mode 1: 1.598e-13 Hz natural, 0 Hz damped, damping ratio 0, largest in v[5], coordinate 5 of the free joint of body 1 (default_sedan_chassis)
mode 2: 4.537e-12 Hz natural, 0 Hz damped, damping ratio 0, largest in v[3], coordinate 3 of the free joint of body 1 (default_sedan_chassis)
mode 3: 5.266e-10 Hz natural, 0 Hz damped, damping ratio 0, largest in v[3], coordinate 3 of the free joint of body 1 (default_sedan_chassis)
mode 4: 0.007636 Hz natural, 0 Hz damped, damping ratio 1, largest in v[3], coordinate 3 of the free joint of body 1 (default_sedan_chassis)
mode 5: 1.04 Hz natural, 0 Hz damped, damping ratio 1, largest in v[26], the revolute joint of body 14 (dyn_UCA)
mode 6: 1.303 Hz natural, 0.9168 Hz damped, damping ratio 0.7104, largest in v[21], the revolute joint of body 11 (dyn_UCA)
mode 7: 1.764 Hz natural, 1.577 Hz damped, damping ratio 0.4482, largest in v[26], the revolute joint of body 14 (dyn_UCA)
mode 8: 4.891 Hz natural, 3.181 Hz damped, damping ratio 0.7597, largest in v[11], the revolute joint of body 5 (dyn_UCA)
mode 9: 7.371 Hz natural, 0 Hz damped, damping ratio 1, largest in v[26], the revolute joint of body 14 (dyn_UCA)
mode 10: 11.77 Hz natural, 8.877 Hz damped, damping ratio 0.6568, largest in v[11], the revolute joint of body 5 (dyn_UCA)
mode 11: 13.46 Hz natural, 11.27 Hz damped, damping ratio 0.5466, largest in v[26], the revolute joint of body 14 (dyn_UCA)
mode 12: 33.18 Hz natural, 0 Hz damped, damping ratio 1, largest in v[21], the revolute joint of body 11 (dyn_UCA)
mode 13: 35.77 Hz natural, 0 Hz damped, damping ratio 1, largest in v[11], the revolute joint of body 5 (dyn_UCA)
mode 14: 47.46 Hz natural, 0 Hz damped, damping ratio 1, largest in v[21], the revolute joint of body 11 (dyn_UCA)
mode 15: 57 Hz natural, 0 Hz damped, damping ratio 1, largest in v[16], the revolute joint of body 8 (dyn_UCA)
undamped 1: 0 Hz, largest in v[1], coordinate 1 of the free joint of body 1 (default_sedan_chassis)
undamped 2: 0 Hz, largest in v[5], coordinate 5 of the free joint of body 1 (default_sedan_chassis)
undamped 3: 0 Hz, largest in v[3], coordinate 3 of the free joint of body 1 (default_sedan_chassis)
undamped 4: 1.149 Hz, largest in v[21], the revolute joint of body 11 (dyn_UCA)
undamped 5: 1.695 Hz, largest in v[11], the revolute joint of body 5 (dyn_UCA)
undamped 6: 1.847 Hz, largest in v[26], the revolute joint of body 14 (dyn_UCA)
undamped 7: 17.6 Hz, largest in v[21], the revolute joint of body 11 (dyn_UCA)
undamped 8: 17.63 Hz, largest in v[21], the revolute joint of body 11 (dyn_UCA)
undamped 9: 17.78 Hz, largest in v[11], the revolute joint of body 5 (dyn_UCA)
undamped 10: 17.79 Hz, largest in v[16], the revolute joint of body 8 (dyn_UCA)
Note: MBD-K081: 3 of the 10 undamped modes have zero frequency: directions without stiffness (a vehicle's position and heading on a flat road). In A they appear as zero or real eigenvalues.
time 5.2 ms
```

The quarter car's statics, then its linearisation: the undamped frequencies
against the closed form, the damped eigenvalues against those of the state
matrix written by hand (`ref eig`), the reduced K, C, M and N (the sliders'
own coordinates here), and B. Then the double-wishbone sedan, settled by
`static_equilibrium`: modes 1 to 3 are its free directions (zero to
roundoff); mode 4 and the real modes are motions damped by the tyres' slip
at low speed and by the dampers; the undamped body modes are at 1.15, 1.70
and 1.85 Hz and the wheel hop near 17.7 Hz.

## The other cases (`lincases`)

Relative errors against each case's closed form or energy method:

```
oscillator: fn 2.79e-16 zeta 8.25e-12 K 0 C 8.25e-12 B 2.22e-16 (relative)
closure pendulum: omega^2 rel err 1.44e-11
four-bar: omega^2 265.326108 energy method 265.3261029 rel err 1.95e-08 (2.592 Hz)
upright: omega^2 rel err 2.26e-11 growth rate rel err 1.13e-11
```

## Programs

```cpp
#include <cstdio>
#include <iostream>
#include <chrono>
#include <Eigen/Geometry>
#include "mbd/kernel/statics.hpp"
#include "mbd/kernel/linearization.hpp"
#include "mbd/forces/tire.hpp"
#include "mbd/vehicle/vehicle_template.hpp"
using namespace mbd; using namespace mbd::kernel;
int main(){
  const Real ms=250, mu=40, ks=20000, kt=200000, cs=1500, R=0.35, L0=0.3;
  System sys; sys.model.gravity=Vec3(0,-g_accel,0);
  const Transform3 zy=Transform3::FromRotation(Mat3(Eigen::AngleAxisd(-pi/2,Vec3::UnitX()).toRotationMatrix()));
  auto pr=std::make_shared<PrismaticJointModel>();
  sys.model.add_body(0,pr,zy,zy,RigidBodyInertia::from_solid_box(mu,Vec3(0.15,0.15,0.15)),"wheel");
  sys.model.add_body(0,pr,zy,zy,RigidBodyInertia::from_solid_box(ms,Vec3(0.5,0.2,0.4)),"chassis");
  sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(1,2,Vec3::Zero(),Vec3::Zero(),ks,cs,L0));
  sys.force_elements.push_back(std::make_shared<TireContactForce>(1,R,kt,0.0));
  Simulator sim(sys); sim.q << R-0.01, R+0.25;
  auto st=static_equilibrium(sim); printf("%s",st.summary(sys.model).c_str());
  auto L=linearize(sim); printf("%s",L.summary(sys.model).c_str());
  const Real a=(ks+kt)/mu, b=ks/ms, sum=a+b, prod=a*b-(ks/mu)*(ks/ms), disc=std::sqrt(sum*sum-4*prod);
  printf("analytic undamped %.12f %.12f Hz\n", std::sqrt((sum-disc)/2)/(2*pi), std::sqrt((sum+disc)/2)/(2*pi));
  for (auto& u: L.undamped) printf("lin undamped %.12f Hz\n", u.frequency_hz);
  { const Real w1=std::sqrt((sum-disc)/2); printf("bounce shape ratio %.12f analytic %.12f err %.3g\n", L.undamped[0].shape(0)/L.undamped[0].shape(1), (ks-w1*w1*ms)/ks, std::abs(L.undamped[0].shape(0)/L.undamped[0].shape(1)-(ks-w1*w1*ms)/ks)); }
  Eigen::Matrix2d M; M<<mu,0,0,ms; Eigen::Matrix2d K; K<<ks+kt,-ks,-ks,ks; Eigen::Matrix2d C; C<<cs,-cs,-cs,cs;
  Eigen::Matrix4d A=Eigen::Matrix4d::Zero(); A.topRightCorner(2,2).setIdentity(); A.bottomLeftCorner(2,2)=-M.inverse()*K; A.bottomRightCorner(2,2)=-M.inverse()*C;
  Eigen::EigenSolver<Eigen::Matrix4d> es(A); std::cout.precision(12); for(int i=0;i<4;++i) printf("ref eig %.12f %+.12f i\n", es.eigenvalues()(i).real(), es.eigenvalues()(i).imag());
  for (auto& m: L.modes) printf("lin eig %.12f %+.12f i  shape (%.6f%+.6fi, %.6f%+.6fi)\n", m.eigenvalue.real(), m.eigenvalue.imag(), m.shape(0).real(), m.shape(0).imag(), m.shape(1).real(), m.shape(1).imag());
  std::cout << "K=\n" << L.K << "\nC=\n" << L.C << "\nM=\n" << L.M << "\nN=\n" << L.N << "\nB=\n" << L.B << std::endl;
  // sedan
  auto tmpl = VehicleTemplate::DefaultSedan(); tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone; tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;
  System s2; auto vh=build_vehicle(s2,tmpl); Simulator sim2(s2); set_vehicle_equilibrium(sim2,vh); sim2.initialize(); static_equilibrium(sim2);
  auto t0=std::chrono::steady_clock::now(); auto L2=linearize(sim2); auto t1=std::chrono::steady_clock::now();
  printf("%s", L2.summary(s2.model).c_str()); printf("time %.1f ms\n", std::chrono::duration<double,std::milli>(t1-t0).count());
}
```

```cpp
#include <cstdio>
#include <cmath>
#include <Eigen/Geometry>
#include "mbd/kernel/statics.hpp"
#include "mbd/kernel/linearization.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/forces/force_element.hpp"
#include "kernel/four_bar.hpp"
using namespace mbd; using namespace mbd::kernel; using mbd_test::FourBar;
Transform3 zy(){ return Transform3::FromRotation(Mat3(Eigen::AngleAxisd(-0.5*pi,Vec3::UnitX()).toRotationMatrix())); }
std::shared_ptr<JointCoordinateForce> ts(const Model& md,int b,Real k,Real r){JointCoordinateForceParams p; p.spring=Curve::linear(k); p.reference=r; return std::make_shared<JointCoordinateForce>(md,b,p);}
Real rel(Real a, Real b){return std::abs(a-b)/std::abs(b);}
int main(){
 { const Real m=2,k=800,c=12,L0=0.5; System sys; sys.model.gravity=Vec3(0,-g_accel,0); int b=sys.model.add_body(0,std::make_shared<PrismaticJointModel>(),zy(),zy(),RigidBodyInertia::from_solid_box(m,Vec3(0.1,0.1,0.1))); sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(0,b,Vec3::Zero(),Vec3::Zero(),k,c,L0)); Simulator sim(sys); sim.q(0)=L0; static_equilibrium(sim); auto L=linearize(sim);
   const Real wn=std::sqrt(k/m), z=c/(2*std::sqrt(k*m)); printf("oscillator: fn %.3g zeta %.3g K %.3g C %.3g B %.3g (relative)\n", rel(L.modes[0].natural_frequency_hz, wn/(2*pi)), rel(L.modes[0].damping_ratio,z), rel(L.K(0,0),k), rel(L.C(0,0),c), rel(L.B(1,0)*L.N(0,0),1/m)); }
 { const Real m=3,d=0.4; System sys; sys.model.gravity=Vec3(0,-g_accel,0); auto I=RigidBodyInertia::from_solid_box(m,Vec3(0.05,0.3,0.08)); int body=sys.model.add_body(0,std::make_shared<FreeJointModel>(),Transform3(),Transform3(),I); sys.constraints.push_back(revolute_closure(Marker{0,Transform3()},Marker{body,Transform3::FromTranslation(Vec3(0,d,0))})); Simulator sim(sys); sim.q.head<3>()=Vec3(0,-d,0); sim.initialize(); auto L=linearize(sim);
   printf("closure pendulum: omega^2 rel err %.3g\n", rel(L.undamped[0].omega_squared, m*g_accel*d/(I.I_com_B(2,2)+m*d*d))); }
 { FourBar fb; System sys; sys.model=fb.model; sys.constraints=fb.closure; const Real k=50,th0=1; sys.joint_forces.push_back(ts(sys.model,fb.crank,k,th0)); Simulator sim(sys); sim.q=FourBar::closed(1.2); static_equilibrium(sim); Real th=sim.q(0); auto L=linearize(sim);
   Data data(sys.model); auto V=[&](Real x){forward_kinematics(sys.model,data,FourBar::closed(x),VecX::Zero(3)); return potential_energy(sys.model,data)+0.5*k*(x-th0)*(x-th0);};
   Real H=1e-4,hq=1e-5; Real V2=(V(th+H)-2*V(th)+V(th-H))/(H*H); VecX dq=(FourBar::closed(th+hq)-FourBar::closed(th-hq))/(2*hq); crba(sys.model,data,sim.q); Real Me=dq.dot(data.M*dq);
   printf("four-bar: omega^2 %.10g energy method %.10g rel err %.3g (%.4g Hz)\n", L.undamped[0].omega_squared, V2/Me, rel(L.undamped[0].omega_squared,V2/Me), L.undamped[0].frequency_hz); }
 { const Real m=1,l=0.5,k=1; System sys; sys.model.gravity=Vec3(0,-g_accel,0); auto I=RigidBodyInertia::from_solid_box(m,Vec3(0.02,0.02,0.02)); I.com_B=Vec3(l,0,0); int b=sys.model.add_body(0,std::make_shared<RevoluteJointModel>(),Transform3(),Transform3(),I); sys.joint_forces.push_back(ts(sys.model,b,k,0.5*pi)); Simulator sim(sys); sim.q(0)=0.5*pi; auto L=linearize(sim);
   Real w2=(k-m*g_accel*l)/(I.I_com_B(2,2)+m*l*l); Real lr=-1e300; for(auto& md: L.modes) lr=std::max(lr, md.eigenvalue.real()); printf("upright: omega^2 rel err %.3g growth rate rel err %.3g\n", rel(L.undamped[0].omega_squared,w2), rel(lr,std::sqrt(-w2))); }
}
```
