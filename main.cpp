#include <iostream>
#include <librealsense2/rs.h>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <GLFW/glfw3.h>

using namespace std;
using namespace cv;

void quat2mat(rs2_quaternion &q, GLfloat H[16]) // to column-major matrix
{
    H[0] = 1 - 2 * q.y * q.y - 2 * q.z * q.z;
    H[4] = 2 * q.x * q.y - 2 * q.z * q.w;
    H[8] = 2 * q.x * q.z + 2 * q.y * q.w;
    H[12] = 0.0f;
    H[1] = 2 * q.x * q.y + 2 * q.z * q.w;
    H[5] = 1 - 2 * q.x * q.x - 2 * q.z * q.z;
    H[9] = 2 * q.y * q.z - 2 * q.x * q.w;
    H[13] = 0.0f;
    H[2] = 2 * q.x * q.z - 2 * q.y * q.w;
    H[6] = 2 * q.y * q.z + 2 * q.x * q.w;
    H[10] = 1 - 2 * q.x * q.x - 2 * q.y * q.y;
    H[14] = 0.0f;
    H[3] = 0.0f;
    H[7] = 0.0f;
    H[11] = 0.0f;
    H[15] = 1.0f;
}

int main(int argc, char *argv[])
{
    // Create a Pipeline - this serves as a top-level API for streaming and processing frames
    rs2::pipeline p;
    rs2::config cfg;
    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;
    // Declare rates printer for showing streaming rates of the enabled streams.
    rs2::rates_printer printer;

    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 30);
    rs2::pipeline_profile profile = p.start(cfg);

    //rgb
    rs2::sensor sensor = profile.get_device().query_sensors()[1]; //0是深度图，1是彩色图
    float max_rgb = sensor.get_option_range(RS2_OPTION_EXPOSURE).max;
    float min_rgb = sensor.get_option_range(RS2_OPTION_EXPOSURE).min;
    cout << max_rgb << ' ' << min_rgb << endl;
    //float exp = 1000;
    // sensor.set_option(RS2_OPTION_EXPOSURE, exp);
    sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);

    //dept
    // Declare pointcloud object, for calculating pointcloud and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops

    rs2::sensor dept_sensor = profile.get_device().query_sensors()[0]; //0是深度图，1是彩色图
    dept_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
    // depth w.r.t. tracking (column-major)
    float H_t265_d400[16] = {1, 0, 0, 0,
                             0, -1, 0, 0,
                             0, 0, -1, 0,
                             0, 0, 0, 1};

    while (1)
    {

        // Block program until frames arrive
        rs2::frameset frames = p.wait_for_frames(). // Wait for next set of frames from the camera
                               apply_filter(printer)
                                   .                    // Print each enabled stream frame rate
                               apply_filter(color_map); // Find and colorize the depth data
        // Try to get a frame
        rs2::video_frame color = frames.get_color_frame();
        if (color.supports_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE))
            cout << color.get_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE) << endl;

        rs2::depth_frame depth = frames.get_depth_frame();
        if (depth.supports_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE))
            cout << depth.get_frame_metadata(RS2_FRAME_METADATA_ACTUAL_EXPOSURE) << endl;

        //dept_calculation
        rs2::points points = pc.calculate(depth);
        pc.map_to(color);

        rs2_pose pose;

        rs2::pose_frame pose_frame = frames.get_pose_frame();
        if (pose_frame)
        {
            pose = pose_frame.get_pose_data();
        }

        /* this segment actually prints the pointcloud */
        auto vertices = points.get_vertices();              // get vertices
        auto tex_coords = points.get_texture_coordinates(); // and texture coordinates
        for (int i = 0; i < points.size(); i++)
        {
            if (vertices[i].z)
            {
                // upload the point and texture coordinates only for points we have depth data for
                glVertex3fv(vertices[i]);
                glTexCoord2fv(tex_coords[i]);
            }
        }

        // viewing matrix

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();

        GLfloat H_world_t265[16];
        quat2mat(pose.rotation, H_world_t265);
        H_world_t265[12] = pose.translation.x;
        H_world_t265[13] = pose.translation.y;
        H_world_t265[14] = pose.translation.z;

        glMultMatrixf(H_world_t265);
        glMultMatrixf(H_t265_d400);

        // Get the depth frame's dimensions
        // const int w = depth.as<rs2::video_frame>().get_width();  //848
        // const int h = depth.as<rs2::video_frame>().get_height(); //480
        // cout << w << ',' << h << endl;

        // float width_depth = depth.get_width();
        // float height_depth = depth.get_height();
        // Query the distance from the camera to the object in the center of the image
        // float dist_to_center = depth.get_distance(width_depth / 2, height_depth / 2);

        // Print the distance
        // std::cout << "The camera is facing an object " << dist_to_center << " meters away \r";

        Mat image(Size(color.get_width(), color.get_height()), CV_8UC3, (void *)color.get_data(), Mat::AUTO_STEP);
        Mat dept_image(Size(depth.get_width(), depth.get_height()), CV_8UC3, (void *)depth.get_data(), Mat::AUTO_STEP);
        imshow("image", image);
        imshow("dept_image", dept_image);

        waitKey(1);
    }
}
