#pragma once

#include <QDebug>
#include <QtWidgets>
#include <Evaluation/batch.h>

//-----------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
	QApplication a( argc, argv );	

	dkeval::runMLDP( argv );

	qInfo() << "PROGRAM FINISHED"; 

	return a.exec();
}

//-------------------------------------------------------------------------------

