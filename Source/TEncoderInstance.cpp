#include "TEncoderInstance.h"


PopH264::TEncoderInstance::TEncoderInstance(const std::string& Encoder)
{
	std::stringstream Error;
	Error << "No encoder supported (requested " << Encoder << ")";
	throw Soy::AssertException(Error);
}


void PopH264::TEncoderInstance::PushFrame(const SoyPixelsImpl& Frame,const std::string& Meta)
{
	Soy_AssertTodo();
}

void PopH264::TEncoderInstance::PeekPacket(json11::Json::object& Meta)
{
	Soy_AssertTodo();
}

size_t PopH264::TEncoderInstance::PeekNextFrameSize()
{
	Soy_AssertTodo();
}

void PopH264::TEncoderInstance::PopPacket(ArrayBridge<uint8_t>&& Data)
{
	Soy_AssertTodo();
}
