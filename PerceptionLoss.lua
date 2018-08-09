require 'nn'
require 'torch'
require 'preprocess'

local PerceptionLoss, parent = torch.class('nn.PerceptionLoss', 'nn.Criterion')

function PerceptionLoss:__init(vgg_model,num)
    parent.__init(self)
	vgg_model      = createVggmodel_trim()
	self.vgg_model = vgg_model
	self.vgg_model:zeroGradParameters()
	self.f_input   = torch.Tensor()
	self.f_target  = torch.Tensor()
	self.output    = 0
	self.gradInput = torch.Tensor()
	self.criterion = nn.MSECriterion()
end
function PerceptionLoss:updateOutput(input, target)
	---[[
	-- what is the range of the loss ? I think I should normalize them.
	local input_vgg  = input
	local target_vgg = target
	
	-- ********* This is very important: ***************
	-- because vgg output a table, and the table is just a reference.
	-- so every time input = target. (This is the reason)
	-- VERY IMPORTANT !!!! ORDER Matters!!!
	self.f_target = self.vgg_model:forward(target_vgg):clone()
	self.f_input  = self.vgg_model:forward(input_vgg):clone()
	
--	self.f_target:div(255):mul(2):add(-1)            --added by Fatima
--	self.f_target:div(255):mul(2):add(-1)            --added by Fatima
	
	self.output = self.criterion:forward(self.f_input, self.f_target)
	
	--]]
--	self.f_input:resizeAs(input):copy(input)
--	self.f_target:resizeAs(target):copy(target)
--	self.output = torch.mean( torch.pow(self.f_input - self.f_target,2))
	--print('Output:  ' .. tostring(self.output))
    return self.output
end

function PerceptionLoss:updateGradInput(input, target)
	---[[
	local input_vgg  = input
	
--	self.f_input  = self.vgg_model:forward(input_vgg:cuda()):clone()
--	self.f_target = self.vgg_model:forward(target_vgg:cuda()):clone()
	
	self.gradInput:resizeAs(input):zero()
	self.vgg_model:zeroGradParameters()
	
	local grad = self.criterion:backward(self.f_input, self.f_target)
	
	-- here matters, i should uncomment this before backforwards
	-- one trick is flip the order in forward of VGG:forward. 
	-- self.f_input = self.vgg_model:forward(input_vgg):clone() 
	self.gradInput = self.vgg_model:backward(input_vgg, grad):clone()
	--self.gradInput:div(input:size(1))
	
	--]]
	
--	local tmp = (input-target)*2
--	self.gradInput:resizeAs(input):copy(tmp):div(input:size(1))
	--print('grad norm: ' .. tostring(self.gradInput:norm()) )
	return self.gradInput
end