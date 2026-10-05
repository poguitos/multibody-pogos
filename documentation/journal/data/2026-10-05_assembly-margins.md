# Assembly margins, task 3.4

- **Date:** 5 October 2026, cloud session (Linux container, GCC 13.3, RelWithDebInfo).
- **Produced by:** the program below, compiled against the library at the
  commit of task 3.4, from the repository root:

  ```sh
  g++ -std=c++20 -O1 -Iinclude -Itests -Ibuild-linux/_deps/eigen-src probe2.cpp \
      build-linux/src/libmultibody_core.a -o probe2 && ./probe2
  ```

## Output

```
assembly: velocities 3; constraint equations 5 (2 independent); degrees of freedom 1
positions: 1 held, 0 degree(s) of freedom left; |phi| 0.2236 -> 3.839e-12 in 4 Gauss-Newton steps (0 towards the least correction); largest change 0.3 in v[2], the revolute joint of body 3 (rocker)
velocities: 1 held, 0 degree(s) of freedom left; |J v - nu| 1.4 -> 2.22e-16; largest change 2.236 in v[1], the revolute joint of body 2 (coupler)
accelerations: 1 held, 0 degree(s) of freedom left; |J a - gamma| 0.9055 -> 1.11e-16; largest change 1.438 in v[1], the revolute joint of body 2 (coupler)
sing 0.627086 0.565541 bound 3.53644e-10
q err 6.29397e-12
v err 1.64649e-11
a err 3.65737e-07  |ddq| 0.359566
|q''''| ~ 1.09661
|q'''| ~ 0.524412
assembly: velocities 3; constraint equations 5 (2 independent); degrees of freedom 1
positions: 0 held, 1 degree(s) of freedom left; |phi| 0.282 -> 1.57e-16 in 23 Gauss-Newton steps (11 towards the least correction); largest change 0.9877 in v[1], the revolute joint of body 2 (coupler)
velocities: 0 held, 1 degree(s) of freedom left; |J v - nu| 0 -> 0; nothing changed
accelerations: not assembled
Note: MBD-K066: The held positions leave 1 degree(s) of freedom; along them the assembly changed the given positions as little as possible, in the kinetic-energy metric.
kkt assembled 5.56264e-11 projection 0.0809412
brute th 1.7234306014 diff 2.41601e-09 cost asm 0.0954606887499 proj 0.107749120009
assembly: velocities 3; constraint equations 5 (2 independent); degrees of freedom 1
positions: 2 held, 0 degree(s) of freedom left; |phi| 0.1198 -> 0.1178 in 7 Gauss-Newton steps (0 towards the least correction); largest change 0.03388 in v[1], the revolute joint of body 2 (coupler); NOT CONVERGED
velocities: not assembled
accelerations: not assembled
Error: MBD-K062: The positions could not be assembled: |phi| is 0.1178 after 7 Gauss-Newton steps (tolerance 1e-10); the largest residual is in constraint 0 (revolute closure), 0.1066. The velocities were left as given.
Warning: MBD-K065: The 2 held positions take 1 of the 2 independent constraint directions away from the coordinates left free: the held values must satisfy 1 constraint equation(s) by themselves.
```

## Program

```cpp
#include <cstdio>
#include <Eigen/Dense>
#include "mbd/kernel/assembly.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "kernel/four_bar.hpp"
using namespace mbd; using namespace mbd::kernel; using mbd_test::FourBar;
int main(){
  FourBar fb; System sys; sys.model=fb.model; sys.constraints=fb.closure;
  // done-when case
  { VecX q = FourBar::closed(1.0)+Eigen::Vector3d(0,0.25,-0.3), v=Eigen::Vector3d(2,0,0), a=VecX::Zero(3);
    AssemblySpec spec; spec.hold(sys.model,1);
    auto r=assemble(sys,q,v,a,0.0,spec);
    printf("%s", r.summary(sys.model).c_str());
    Data data(sys.model); ConstraintSolver s(sys.model, sys.constraints); s.evaluate(data,q,VecX::Zero(3),0.0);
    MatX Jf(5,2); Jf.col(0)=s.J().col(1); Jf.col(1)=s.J().col(2);
    Eigen::JacobiSVD<MatX> svd(Jf); printf("sing %g %g bound %g\n", svd.singularValues()(0), svd.singularValues()(1), 2e-10/svd.singularValues()(1));
    printf("q err %g\n", (q-FourBar::closed(1.0)).cwiseAbs().maxCoeff());
    double h=1e-5; VecX dq=(FourBar::closed(1+h)-FourBar::closed(1-h))/(2*h); printf("v err %g\n",(v-2*dq).cwiseAbs().maxCoeff());
    double H=1e-3; VecX ddq=(FourBar::closed(1+H)-2*FourBar::closed(1)+FourBar::closed(1-H))/(H*H); printf("a err %g  |ddq| %g\n",(a-4*ddq).cwiseAbs().maxCoeff(), ddq.cwiseAbs().maxCoeff());
    // estimate q'''' magnitude
    double H2=1e-2; VecX d4=(FourBar::closed(1+2*H2)-4*FourBar::closed(1+H2)+6*FourBar::closed(1)-4*FourBar::closed(1-H2)+FourBar::closed(1-2*H2))/std::pow(H2,4); printf("|q''''| ~ %g\n", d4.cwiseAbs().maxCoeff());
    VecX d3=(FourBar::closed(1+2*H2)-2*FourBar::closed(1+H2)+2*FourBar::closed(1-H2)-FourBar::closed(1-2*H2))/(2*std::pow(H2,3)); printf("|q'''| ~ %g\n", d3.cwiseAbs().maxCoeff());
  }
  // least correction
  { VecX qg = FourBar::closed(1.0)+Eigen::Vector3d(0.1,0.25,-0.3); Data data(sys.model); crba(sys.model,data,qg); MatX M0=data.M;
    auto opt=[&](const VecX& at){ ConstraintSolver s(sys.model, sys.constraints); Data d2(sys.model); s.evaluate(d2,at,VecX::Zero(3),0.0); Eigen::FullPivLU<MatX> lu(s.J()); VecX n=lu.kernel().col(0).normalized(); VecX d=at-qg; return std::abs(n.dot(M0*d))/(M0*d).norm(); };
    VecX q=qg, v=VecX::Zero(3); auto r=assemble(sys,q,v,0.0); printf("%s", r.summary(sys.model).c_str());
    VecX qp=qg, vp=v; ConstraintSolver s(sys.model, sys.constraints); s.project(data,qp,vp,0.0);
    printf("kkt assembled %g projection %g\n", opt(q), opt(qp));
    auto cost=[&](double th){VecX d=FourBar::closed(th)-qg; return d.dot(M0*d);};
    double lo=0.5,hi=2.5,g=0.5*(std::sqrt(5.0)-1); for(int k=0;k<200;++k){double x1=hi-g*(hi-lo),x2=lo+g*(hi-lo); if(cost(x1)<cost(x2)) hi=x2; else lo=x1;}
    VecX qb=FourBar::closed(0.5*(lo+hi)); printf("brute th %.10f diff %g cost asm %.12g proj %.12g\n",0.5*(lo+hi),(q-qb).cwiseAbs().maxCoeff(), cost(q(0)), (qp-qg).dot(M0*(qp-qg)));
  }
  // conflicting
  { VecX q = FourBar::closed(1.0)+Eigen::Vector3d(0,0,0.2), v=VecX::Zero(3); AssemblySpec spec; spec.hold_positions={0,2};
    auto r=assemble(sys,q,v,0.0,spec); printf("%s", r.summary(sys.model).c_str()); }
}
```

## Reading it

- First block: the four-bar with its crank held (the test "a four-bar given
  inconsistent values is assembled and reported"). `sing` are the singular
  values of the coupler and rocker columns of J; `bound` is the test's bound
  on the position error, 2 x tolerance / smallest. `|q'''|` and `|q''''|` are
  estimated by finite differences with step 0.01, to size the truncation
  errors of the velocity and acceleration checks.
- Second block: nothing held. `kkt` is |n . M0 d| / |M0 d|, n the loop's
  tangent at the result; `brute` is the golden-section search along the loop;
  `cost` is d^T M0 d for the assembly and for a plain projection.
- Third block: crank and rocker held at values that contradict the loop. The
  least-squares fallback brings |phi| from 0.1198 to 0.1178, the least the
  coupler alone can reach, and the report says why.
