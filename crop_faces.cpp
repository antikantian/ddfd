#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <regex>

using namespace dlib;
using namespace std;

struct image_info
{
	string filename;
	rectangle bbox;
	string savename;
};

std::vector<image_info> get_image_listing(
		const std::string& images_folder
)
{
	std::vector<image_info> results;
	image_info temp;

	std::regex file_regex("(\\d+-\\d+-[\\w.]+)-\\((\\d+)x(\\d+)x(\\d+)x(\\d+)\\).jpg");
	std::smatch file_match;

	auto dir = directory(images_folder);

	for (auto image_file : dir.get_files()) {
		if (std::regex_match(image_file.name(), file_match, file_regex)) {
			if (file_match.size() == 6) {
				temp.bbox = rectangle(stol(file_match[2].str()), stol(file_match[3].str()), stol(file_match[4].str()), stol(file_match[5].str()));
				temp.filename = image_file;

				ostringstream saves;
				saves << file_match[1].str() << ".jpg";
				temp.savename = saves.str();

				results.push_back(temp);
			}
		}
	}

	return results;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	try
	{
		if (argc != 3)
		{
			cout << "Usage: ./crop_faces /path/to/images /path/to/save/images" << endl;
			return 0;
		}

		cout << "\nSCANNING IMAGES\n" << endl;

		auto images = get_image_listing(string(argv[1]));
		cout << "number of images: " << images.size() << endl;

		auto outpath = argv[2];

		array2d<rgb_pixel> img;
		array2d<rgb_pixel> crop;

		for (int i = 0; i < images.size(); ++i)
		{
			cout << "(" << images[i].savename << ") processing image " << i << " of " << images.size() << endl;

			try {
				load_image(img, images[i].filename);
				extract_image_chip(img, chip_details(images[i].bbox), crop);

				ostringstream outstream;
				outstream << outpath << images[i].savename;

				save_jpeg(crop, outstream.str());
			} catch (...) { }
		}
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}