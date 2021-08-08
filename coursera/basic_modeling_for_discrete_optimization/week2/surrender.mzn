% workshop 2
% select a set of trade negotiation party
enum NEGOTIATOR;%= { D, N1, N2, N3, N4, N5, N6, N7 };
NEGOTIATOR: dummy;% = D; 

int: l;% = 3; % minimum party size
int: u;% = 4; % maximum party size
int: m;% = 40; % minimum joint ability

array[NEGOTIATOR] of int: honor;% = [ 0, 10, 8, 6, 3, 5, 7, 3 ];
array[NEGOTIATOR,NEGOTIATOR] of int: joint;% = 
/*
        [| 0, 0, 0, 0, 0, 0, 0, 0
         | 0, 4, 7, 8, 4, 2, 9, 7
         | 0, 7, 9, 3, 1, 8, 6, 6
         | 0, 8, 3, 5, 2, 9, 2, 9
         | 0, 4, 1, 2, 8, 2, 4, 1
         | 0, 2, 8, 9, 2, 7, 2, 5
         | 0, 9, 6, 2, 4, 2, 1, 9,
         | 0, 7, 6, 9, 1, 5, 9, 4 |];
*/
%array[1..u] of var NEGOTIATOR;


var set of NEGOTIATOR: party;
constraint card(party) >= l;
constraint card(party) <= u;
constraint sum(i, j in party where i<j)(joint[i,j]) >= m;
var int: obj;
constraint obj = min(i in party)(honor[i]);

solve maximize obj;
output[
"party = \(party);\nobj = \(obj);\n"];
