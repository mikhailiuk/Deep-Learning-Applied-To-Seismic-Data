function []=videoWeights(size, path, counter, name)
    if size>1
            %Create and set a videowriter
            v = VideoWriter(char(strcat(path,'/weights_video_',name,'_',num2str(counter),'.avi')));
            v.FrameRate = 2;
            open(v);

            %For every frame (weight) in Z direction (from top to bottow)
            for frame  = 1:size

                %Write the frame
                im = imread(char(strcat(path,'weights_',name,'_',int2str(counter),'_',int2str(frame),'.png')));
                writeVideo(v,im);
            end

            %Close the video
            close(v);       
    end
end