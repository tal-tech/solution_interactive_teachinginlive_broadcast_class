#ifndef __FACETHINK_API_FACE_DETECTION_CONFIG_HPP__
#define __FACETHINK_API_FACE_DETECTION_CONFIG_HPP__

#include <vector>
#include <string>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

namespace facethink {
	namespace detfacemtcnn {

		class Config {
		public:
			Config() {

				SCALE = 0.5;
				MINSIZE = 30.0;
				FASE_RESIZE = false;
				FACTOR = 0.75;
				THRESHOLD[0] = 0.75;
				THRESHOLD[1] = 0.85;
				THRESHOLD[2] = 0.85;
				THRESHOLD_NMS = 0.7;

				boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::error);

			}

		public:

			float SCALE;
			float MINSIZE;
			bool FASE_RESIZE;
			float FACTOR;
			float THRESHOLD[3];
			float THRESHOLD_NMS;

			void ReadIniFile(const std::string& config_file); // class Config
		};
	}
}
#endif
