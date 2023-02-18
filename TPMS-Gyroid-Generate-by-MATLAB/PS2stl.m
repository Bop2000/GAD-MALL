function PS2stl
%Save periodic surface to STL file and visualize
%
% Inspired by Sven Holcombe
% Personal email: gauss@zju.edu.cn
%
%% uncomment one function to build periodic surface
% ps 1
 f = @(x,y,z) cos(pi*x)+cos(pi*y)+cos(pi*z);

% ps 2
% f = @(x,y,z) cos(pi*x).*cos(pi*y).*cos(pi*z) - sin(pi*x).*sin(pi*y).*sin(pi*z); 

% ps 3
% f = @(x,y,z) sin(pi*x).*cos(pi*y) + sin(pi*y).*cos(pi*z) + sin(pi*z).*cos(pi*x); 

% ps 4
%f = @(x,y,z) 2*cos(pi*x).*cos(pi*y) + 2*cos(pi*y).*cos(pi*z) + 2*cos(pi*x).*cos(pi*z) ...
%    - cos(2*pi*x) - cos(2*pi*y) - cos(2*pi*z);

% sphere
%f = @(x,y,z) x.^2 + y.^2 +z.^2 - 1;

% two parallel surface
%f = @(x,y,z) z.^2 - 0.4;

% cylinder
% f = @(x,y,z) x.^2 + y.^2 - 0.4;

% example
%f = @(x,y,z) 1000*x.*y.*z.*log(1+100*(x.^2+y.^2+z.^2))-10;

% one through hole
%k = 1;
%f = @(x,y,z) x.^(2*k)/1.2 + y.^2/0.25 + z.^2/0.25 - 1;

t = -1:.05:1;
[x,y,z] = meshgrid(t);
v = f(x,y,z);
fv = isosurface(x,y,z,v,0);
fvc = isocaps(x,y,z,v,0);

%% save as xx.stl in the workspace
% two STL file combines to form one PS by the software as COMSOL, Solidworks, etc.
stlwrite('PSsurface.stl',fv);
stlwrite('PScaps.stl',fvc);

%% visulize
h = patch(isosurface(x,y,z,v,0)); 
hc = patch(isocaps(x,y,z,v,0));
isonormals(x,y,z,v,h)              
set(h,'FaceColor','y','EdgeColor','none');
set(hc,'FaceColor','y','EdgeColor','none');
xlabel('x');ylabel('y');zlabel('z'); 
alpha(1)
grid off; view([1,1,1]); axis equal; camlight; lighting gouraud  % [-1 1 -1 1 -1 1], equal
