# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.5.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.5.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full"

# Include any dependencies generated for this target.
include CMakeFiles/L-BFGS-full.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/L-BFGS-full.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/L-BFGS-full.dir/flags.make

CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o: CMakeFiles/L-BFGS-full.dir/flags.make
CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o: L-BFGS-full.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o -c "/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full/L-BFGS-full.cpp"

CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full/L-BFGS-full.cpp" > CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.i

CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full/L-BFGS-full.cpp" -o CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.s

CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o.requires:

.PHONY : CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o.requires

CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o.provides: CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o.requires
	$(MAKE) -f CMakeFiles/L-BFGS-full.dir/build.make CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o.provides.build
.PHONY : CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o.provides

CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o.provides.build: CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o


# Object files for target L-BFGS-full
L__BFGS__full_OBJECTS = \
"CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o"

# External object files for target L-BFGS-full
L__BFGS__full_EXTERNAL_OBJECTS =

L-BFGS-full: CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o
L-BFGS-full: CMakeFiles/L-BFGS-full.dir/build.make
L-BFGS-full: CMakeFiles/L-BFGS-full.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable L-BFGS-full"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/L-BFGS-full.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/L-BFGS-full.dir/build: L-BFGS-full

.PHONY : CMakeFiles/L-BFGS-full.dir/build

CMakeFiles/L-BFGS-full.dir/requires: CMakeFiles/L-BFGS-full.dir/L-BFGS-full.cpp.o.requires

.PHONY : CMakeFiles/L-BFGS-full.dir/requires

CMakeFiles/L-BFGS-full.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/L-BFGS-full.dir/cmake_clean.cmake
.PHONY : CMakeFiles/L-BFGS-full.dir/clean

CMakeFiles/L-BFGS-full.dir/depend:
	cd "/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full" "/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full" "/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full" "/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full" "/Users/ken/Desktop/research/External Library/L-BFGS/l-bfgs full/CMakeFiles/L-BFGS-full.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/L-BFGS-full.dir/depend
