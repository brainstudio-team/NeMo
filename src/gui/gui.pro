# QT += network
TEMPLATE = app
CONFIG += qt4 release
MOC_DIR = moc
OBJECTS_DIR = obj
HEADERS += \
	MainWindow.hpp \
	FiringPlot.hpp
SOURCES += \
	MainWindow.cpp \
	FiringPlot.cpp \
	main.cpp
TARGET = plot-firing
