#include "config.hpp"

namespace facethink {
	namespace detfacemtcnn {

		void Config::ReadIniFile(const std::string& config_file) {

			boost::property_tree::ptree pt;
			boost::property_tree::ini_parser::read_ini(config_file, pt);

			SCALE = pt.get<float>("scale", SCALE);
			MINSIZE = pt.get<float>("minsize", MINSIZE);
			FASE_RESIZE = pt.get<bool>("fase_resize", FASE_RESIZE);
			FACTOR = pt.get<float>("factor", FACTOR);
			THRESHOLD[0] = pt.get<float>("threshold0", THRESHOLD[0]);
			THRESHOLD[1] = pt.get<float>("threshold1", THRESHOLD[1]);
			THRESHOLD[2] = pt.get<float>("threshold2", THRESHOLD[2]);
			THRESHOLD_NMS = pt.get<float>("threshold_nms", THRESHOLD_NMS);

			bool log_save = pt.get<bool>("log_save", true);
			if (log_save)
			{
				boost::log::add_file_log
				(
					boost::log::keywords::auto_flush = true,
					boost::log::keywords::file_name = "det_face_mtcnn.log",
					boost::log::keywords::format = "[%TimeStamp%]: %Message%"
				);
				boost::log::add_common_attributes();
			}

			int log_level = pt.get<int>("log_level", 4);
			boost::log::core::get()->set_filter(boost::log::trivial::severity >= log_level);
		}

	}
}