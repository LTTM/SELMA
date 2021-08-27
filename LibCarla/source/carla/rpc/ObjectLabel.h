// Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/MsgPack.h"

#include <cstdint>

namespace carla {
namespace rpc {

/*
  enum class CityObjectLabel : uint8_t {
    None         =   0u,
    Buildings    =   1u,
    Fences       =   2u,
    Other        =   3u,
    Pedestrians  =   4u,
    Poles        =   5u,
    RoadLines    =   6u,
    Roads        =   7u,
    Sidewalks    =   8u,
    Vegetation   =   9u,
    Vehicles     =  10u,
    Walls        =  11u,
    TrafficSigns =  12u,
    Sky          =  13u,
    Ground       =  14u,
    Bridge       =  15u,
    RailTrack    =  16u,
    GuardRail    =  17u,
    TrafficLight =  18u,
    Static       =  19u,
    Dynamic      =  20u,
    Water        =  21u,
    Terrain      =  22u,
    Any          =  0xFF
  };
*/

  enum class CityObjectLabel : uint8_t {
    None         =   0u,
    Buildings    =   1u,
    Fences       =   2u,
    Other        =   3u,
    Pedestrians  =   4u,
    Poles        =   5u,
    RoadLines    =   6u,
    Roads        =   7u,
    Sidewalks    =   8u,
    Vegetation   =   9u,
    Vehicles     =  10u,
    Walls        =  11u,
    TrafficSigns =  12u,
    Sky          =  13u,
    Ground       =  14u,
    Bridge       =  15u,
    RailTrack    =  16u,
    GuardRail    =  17u,
    TrafficLight =  18u,
    Static       =  19u,
    Dynamic      =  20u,
    Water        =  21u,
    Terrain      =  22u,
	Persons      =  40u,
	Riders       =  41u,
	Cars         =  100u,
	Trucks       =  101u,
	Busses       =  102u,
	Trains       =  103u,
	Motorcycles  =  104u,
	Bycicles     =  105u,
    Any          =  0xFF
  };

/* Does not work as intended, some classes are hard-coded
enum class CityObjectLabel : uint8_t {
	
	// CityScapes Classes
	Roads        =   0u,
	Sidewalks    =   1u,
    Buildings    =   2u,
	Walls        =   3u,
    Fences       =   4u,
    Poles        =   5u,
    TrafficLight =   6u,
    TrafficSigns =   7u,
    Vegetation   =   8u,
    Terrain      =   9u,
    Sky          =  10u,
    Pedestrians  = 0xFE, // Class OverWritten at actor spawn
	Persons      =  12u,
	Riders       =  13u,
	Vehicles     = 0xFD, // Class OverWritten at actor spawn
	Cars         =  13u,
	Trucks       =  14u,
	Busses       =  15u,
	Trains       =  16u,
	Motorbikes   =  17u,
	Bycicles     =  18u,
	
	// Additional Carla Classes
	Other        =  19u,
    RoadLines    =  20u,
    Ground       =  21u,
    Bridge       =  22u,
    RailTrack    =  23u,
    GuardRail    =  24u,
    Static       =  25u,
    Dynamic      =  26u,
    Water        =  27u,

	// Void Classes
	None         =  0xFF,
    Any          =  0xFF
  };
  
*/

} // namespace rpc
} // namespace carla

MSGPACK_ADD_ENUM(carla::rpc::CityObjectLabel);
