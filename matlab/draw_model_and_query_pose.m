function draw_model_and_query_pose(X, T_m2q, K, ...
    lookat, ...
    lookfrom, ...
    point_size, ...
    frame_size, ...
    c)

    %           X: Point cloud model of [shape (3 or 4)xN].
    %       T_m2q: Transformation from model to query camera coordinates (e.g. as obtained from OpenCV's solvePnP).
    %           K: Intrinsic matrix for the virtual 'figure camera'.
    % lookat|from: The viewing target and origin of the virtual figure camera.
    %  point_size: Radius of a point (in pixels) that is 1 unit away. (Points further away will appear smaller.)
    %  frame_size: The length (in model units) of the camera and coordinate frame axes.
    %           c: Color associated with each point in X [shape Nx3].

    if size(X, 1) == 3
        X = [X ; ones(1, size(X, 1))];
    else
        X = X./X(3,:);
    end

    if isempty(c)
        c = X(2,:)';
    elseif max(c, 'all') > 1
        c = c / 256.0;
    end

    % Create transformation from model to 'figure camera'
    T_f2m = eye(4);
    T_f2m(1:3,3) = (lookat - lookfrom);
    T_f2m(1:3,3) = T_f2m(1:3,3)/norm(T_f2m(1:3,3));
    T_f2m(1:3,1) = cross([0 1 0]', T_f2m(1:3,3));
    T_f2m(1:3,1) = T_f2m(1:3,1)/norm(T_f2m(1:3,1));
    T_f2m(1:3,2) = cross(T_f2m(1:3,3), T_f2m(1:3,1));
    T_f2m(1:3,2) = T_f2m(1:3,2)/norm(T_f2m(1:3,2));
    T_f2m(1:3,4) = lookfrom;
    T_m2f = inv(T_f2m);

    % Transform point cloud model into 'figure camera' frame
    X_f = T_m2f*X;
    visible = X_f(3,:) > 0;
    not_visible = X_f(3,:) < 0;
    X_f = X_f(:,visible);
    if sum(not_visible) > sum(visible)
        fprintf('[draw_model_and_query_pose] Most of point cloud is behind camera, is that intentional?\n');
    end

    % Project point cloud with depth sorting
    u = project(K, X_f);
    [~,i] = sort(-X_f(3,:));
    scatter(u(1,i), u(2,i), (point_size^2)./X_f(3,:), c(i,:), 'filled');
    hold on;

    % Project the coordinate frame axes of the localized query camera
    T_q2m = inv(T_m2q);
    draw_camera_pose(K, T_m2f*T_q2m, frame_size);

    box on;
    axis equal;
    axis ij;
    xlim([0 K(1,3)*2]); % Arbitrarily set axis limits to 2*principal point
    ylim([0 K(2,3)*2]);
end

function draw_camera_pose(K, T, scale)
    % Draw the axes of T and a pyramid, representing the camera.
    s = scale;
    X = [
        0,0,0,1 ;
        -s,-s,1.5*s,1 ;
        +s,-s,1.5*s,1 ;
        +s,+s,1.5*s,1 ;
        -s,+s,1.5*s,1 ;
        5.0*s,0,0,1 ;
        0,5.0*s,0,1 ;
        0,0,5.0*s,1 ;
        ]';
    uv = project(K, T*X);
    u = uv(1,:);
    v = uv(2,:);
    plot([u(1), u(6)], [v(1), v(6)], 'color', '#ff5555');
    plot([u(1), u(7)], [v(1), v(7)], 'color', '#33cc55');
    plot([u(1), u(8)], [v(1), v(8)], 'color', '#44aaff');
    lines = [1,2, 1,3, 1,4, 1,5, 2,3, 3,4, 4,5, 5,2];
    for j=0:numel(lines)/2-1
        i1 = lines(2*j+1);
        i2 = lines(2*j+2);
        plot([u(i1), u(i2)], [v(i1), v(i2)], 'color', 'k');
    end
end
