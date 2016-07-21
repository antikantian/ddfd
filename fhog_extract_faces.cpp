#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/dir_nav.h>
#include <dlib/threads.h>
#include <iostream>
#include <string>

using namespace dlib;
using namespace std;

thread_pool tp(8);

struct image_info
{
	string filename;
};

std::vector<image_info> get_image_listing(
		const std::string& images_folder
)
{
	std::vector<image_info> results;
	image_info temp;

	auto dir = directory(images_folder);

	for (auto image_file : dir.get_files()) {
		temp.filename = image_file;
		results.push_back(temp);
	}

	return results;
}

void crop_face(
		shape_predictor& sp,
		const image_info& info,
		const string outpath,
		unsigned long crop_size,
		double padding
) {

	array2d<rgb_pixel> img;
	frontal_face_detector detector = get_frontal_face_detector();

	try {
		load_image(img, info.filename);

		std::vector<rectangle> dets;
		dets = detector(img, 1);

		if (dets.size() == 0) {
			int up_count = 0;

			while ((dets.size() == 0) && (up_count < 2)) {
				pyramid_up(img);
				dets = detector(img, 1);
				up_count++;
			}
		}

		if (dets.size() > 0) {
			std::vector<full_object_detection> shapes;
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				full_object_detection shape = sp(img, dets[j]);
				shapes.push_back(shape);
			}

			dlib::array<array2d<rgb_pixel>> face_chips;
			extract_image_chips(img, get_face_chip_details(shapes, crop_size, padding), face_chips);

			for (int j = 0; j < face_chips.size(); ++j) {
				ostringstream opath;

				dlib::file filename = dlib::file(info.filename);

				if (face_chips.size() == 1) {
					opath << outpath << filename.name();
				} else {
					opath << outpath << j << "-" << filename.name();
				}

				try {
					save_jpeg(face_chips[j], opath.str());
				} catch (dlib::image_save_error& e) {

				}
			}
		}
	} catch (dlib::image_load_error& e) {

	}
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	try
	{
		if (argc != 6)
		{
			cout << "Usage: ./fhog_extract_faces /path/to/shape_predictor.dat /path/to/images /path/to/save crop_size padding" << endl;
			return 0;
		}

		cout << "\nSCANNING IMAGES\n" << endl;

		auto images = get_image_listing(string(argv[2]));
		cout << "number of images: " << images.size() << endl;

		shape_predictor sp;
		deserialize(argv[1]) >> sp;

		auto outpath = argv[3];
		unsigned long crop_size = stoul(argv[4]);
		double padding = stod(argv[5]);

		for (int i = 0; i < images.size(); ++i)
		{
			cout << "processing image " << i << " of " << images.size() << endl;
			tp.add_task_by_value([&sp, &images, i, outpath, crop_size, padding]() {
				crop_face(sp, images[i], outpath, crop_size, padding);
			});
		}
		tp.wait_for_all_tasks();
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}