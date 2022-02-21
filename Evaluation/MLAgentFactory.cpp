#include <Evaluation/MLAgentFactory.h>
#include <FileIo/TabularDataFileIo.h>
#include <QSettings>
#include <QDebug>
#include <Evaluation/MLAgent.h>


namespace lpmleval
{

//-----------------------------------------------------------------------------

MLAgentFactory::MLAgentFactory()
{
}

//-----------------------------------------------------------------------------

MLAgentFactory::~MLAgentFactory()
{
}

//-----------------------------------------------------------------------------

MLAgent* MLAgentFactory::generate( QString aProjectFolder )
{
	QSettings* settings = nullptr;

	// Load up the MLJob setting file from the ProjectFodler.
	QString settingFile = aProjectFolder + "/settings.ini";

	if ( !QFile( settingFile ).exists() )
	{
		qDebug() << "ERROR - Setting file: " << settingFile + " not found./n";
		return nullptr;
	}
	else
	{
		settings = new QSettings( settingFile, QSettings::IniFormat );
		if ( settings != nullptr )
		{
			return new lpmleval::MLAgent( settings );
		}
		else
		{
			qDebug() << "ERROR - settings object is invalid";
			return nullptr;
		}
	}
}

//-----------------------------------------------------------------------------

}
