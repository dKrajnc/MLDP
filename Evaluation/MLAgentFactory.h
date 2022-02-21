#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/MLAgent.h>
#include <QSettings>
#include <QString>

namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API MLAgentFactory
{

public:

	MLAgentFactory();

	virtual ~MLAgentFactory();

	MLAgent* generate( QString aProjectFolder );

private:

private:
};

//-----------------------------------------------------------------------------

}
