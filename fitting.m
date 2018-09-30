x = xy(:,1);
y = xy(:,2);

% scatter3(xy(:,1), xy(:, 2), z)

[fitresult, gof] = createFit(x,y,z);

[xData, yData, zData] = prepareSurfaceData( x, y, z );

figure()
plot(fitresult,[xData, yData], zData,'Style','Residuals')

xlabel x
ylabel y
zlabel z
grid on
view( 14.5, 3.6 );

