int: n = 9;
array[1..n] of var 0..3: x;
constraint x[1] == 0;
constraint x[n-1] == 3;
constraint forall(i in 1..n-1)(x[i] + x[i+1] <=3);
constraint forall(i in 1..n-1)(if i mod 3 == 0 then x[i] >= 2 else true endif);
solve maximize sum(x);
output [show(x)];
