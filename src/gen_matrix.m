B11 = rand(4, 3);
B12 = rand(4, 3);
B13 = rand(4, 3);
B22 = rand(4, 3);
B32 = rand(4, 3);
B34 = rand(4, 3);
B43 = rand(4, 3);
B44 = rand(4, 3);
B52 = rand(4, 3);
B54 = rand(4, 3);
B55 = rand(4, 3);
D11 = rand(4, 4);
D22 = rand(4, 4);
D33 = rand(4, 4);
D44 = rand(4, 4);
D55 = rand(4, 4);
C11 = rand(4, 3);
C12 = rand(4, 3);
C13 = rand(4, 3);
C22 = rand(4, 3);
C32 = rand(4, 3);
C34 = rand(4, 3);
C43 = rand(4, 3);
C44 = rand(4, 3);
C52 = rand(4, 3);
C54 = rand(4, 3);
C55 = rand(4, 3);

Z = zeros(5);

Z(1, 1) = 1;
B = kron(Z, B11);
D = kron(Z, D11);
C = kron(Z, C11);
Z(1, 1) = 0;

Z(1, 2) = 1;
B = B + kron(Z, B12);
C = C + kron(Z, C12);
Z(1, 2) = 0;

Z(1, 3) = 1;
B = B + kron(Z, B13);
C = C + kron(Z, C13);
Z(1, 3) = 0;

Z(2, 2) = 1;
B = B + kron(Z, B22);
D = D + kron(Z, D22);
C = C + kron(Z, C22);
Z(2, 2) = 0;

Z(3, 2) = 1;
B = B + kron(Z, B32);
C = C + kron(Z, C32);
Z(3, 2) = 0;

Z(3, 3) = 1;
D = D + kron(Z, D33);
Z(3, 3) = 0;

Z(3, 4) = 1;
B = B + kron(Z, B34);
C = C + kron(Z, C34);
Z(3, 4) = 0;

Z(4, 3) = 1;
B = B + kron(Z, B43);
C = C + kron(Z, C43);
Z(4, 3) = 0;

Z(4, 4) = 1;
B = B + kron(Z, B44);
D = D + kron(Z, D44);
C = C + kron(Z, C44);
Z(4, 4) = 0;

Z(5, 2) = 1;
B = B + kron(Z, B52);
C = C + kron(Z, C52);
Z(5, 2) = 0;

Z(5, 4) = 1;
B = B + kron(Z, B54);
C = C + kron(Z, C54);
Z(5, 4) = 0;

Z(5, 5) = 1;
B = B + kron(Z, B55);
D = D + kron(Z, D55);
C = C + kron(Z, C55);

B11 = B11';
B11_vals = B11(:);

B12 = B12';
B12_vals = B12(:);

B13 = B13';
B13_vals = B13(:);

B22 = B22';
B22_vals = B22(:);

B32 = B32';
B32_vals = B32(:);

B34 = B34';
B34_vals = B34(:);

B43 = B43';
B43_vals = B43(:);

B44 = B44';
B44_vals = B44(:);

B52 = B52';
B52_vals = B52(:);

B54 = B54';
B54_vals = B54(:);

B55 = B55';
B55_vals = B55(:);

B_vals = [B11_vals; B12_vals; B13_vals; B22_vals;
          B32_vals; B34_vals; B43_vals; B44_vals;
          B52_vals; B54_vals; B55_vals];
      
D11 = D11';
D11_vals = D11(:);

D22 = D22';
D22_vals = D22(:);

D33 = D33';
D33_vals = D33(:);

D44 = D44';
D44_vals = D44(:);

D55 = D55';
D55_vals = D55(:);

D_vals = [D11_vals; D22_vals; D33_vals;
          D44_vals; D55_vals];
      
C11 = C11';
C11_vals = C11(:);

C12 = C12';
C12_vals = C12(:);

C13 = C13';
C13_vals = C13(:);

C22 = C22';
C22_vals = C22(:);

C32 = C32';
C32_vals = C32(:);

C34 = C34';
C34_vals = C34(:);

C43 = C43';
C43_vals = C43(:);

C44 = C44';
C44_vals = C44(:);

C52 = C52';
C52_vals = C52(:);

C54 = C54';
C54_vals = C54(:);

C55 = C55';
C55_vals = C55(:);

C_vals = [C11_vals; C12_vals; C13_vals; C22_vals;
          C32_vals; C34_vals; C43_vals; C44_vals;
          C52_vals; C54_vals; C55_vals];

x = rand(15, 1);
y_middle = D*B*x;
y = C'*D*B*x;

writematrix(B_vals, '../data/valsB.txt');
writematrix(D_vals, '../data/valsD.txt');
writematrix(C_vals, '../data/valsC.txt');
writematrix(x, '../data/x.txt');
writematrix(y_middle, '../data/y_middle.txt');
writematrix(y, '../data/y.txt');
