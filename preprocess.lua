require 'torch'
require 'cunn'
require 'nn'
require 'cudnn'
require 'nngraph'
require 'loadcaffe'

local vgg_mean = {103.939, 116.779, 123.68}

function preprocess_image(img)
	nb = img:size(1)
	local mean = img.new(vgg_mean):view(1, 3, 1, 1):expandAs(img)
	return img:add(1):mul(127.5):add(-1, mean)
end

function createVggmodel()
	input = nn.Identity()()
	conv1_1 = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1)(input)   -- 2
	relu1_1 = cudnn.ReLU(true)(conv1_1)
	conv1_2 = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(relu1_1)
	relu1_2 = cudnn.ReLU(true)(conv1_2)
	pool1   = cudnn.SpatialMaxPooling(2, 2, 2, 2)(relu1_2)
	
	conv2_1 = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)(pool1)  -- 7
	relu2_1 = cudnn.ReLU(true)(conv2_1)
	conv2_2 = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)(relu2_1)
	relu2_2 = cudnn.ReLU(true)(conv2_2)
	pool2   = cudnn.SpatialMaxPooling(2, 2, 2, 2)(relu2_2)
	
	conv3_1 = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)(pool2)  -- 12
	relu3_1 = cudnn.ReLU(true)(conv3_1)
	conv3_2 = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(relu3_1)
	relu3_2 = cudnn.ReLU(true)(conv3_2)
--	conv3_3 = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(relu3_2)
--	relu3_3 = cudnn.ReLU(true)(conv3_3)
--	conv3_4 = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(relu3_3)
--	relu3_4 = cudnn.ReLU(true)(conv3_4)
--	pool3   = cudnn.SpatialMaxPooling(2, 2, 2, 2)(relu3_4)
	
--	conv4_1 = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)(pool3)  -- 21
--	relu4_1 = cudnn.ReLU(true)(conv4_1) 
--	conv4_2 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu4_1)
--	relu4_2 = cudnn.ReLU(true)(conv4_2)
--	conv4_3 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu4_2)
--	relu4_3 = cudnn.ReLU(true)(conv4_3)
--	conv4_4 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu4_3)
--	relu4_4 = cudnn.ReLU(true)(conv4_4)
--	pool4   = cudnn.SpatialMaxPooling(2, 2, 2, 2)(relu4_4)
	
--	conv5_1 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(pool4)   -- 30
--	relu5_1 = cudnn.ReLU(true)(conv5_1)
--	conv5_2 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu5_1)
--	relu5_2 = cudnn.ReLU(true)(conv5_2)
--	conv5_3 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu5_2)
--	relu5_3 = cudnn.ReLU(true)(conv5_3)
--	conv5_4 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(relu5_3)
--	relu5_4 = cudnn.ReLU(true)(conv5_4)
--	pool5   = cudnn.SpatialMaxPooling(2, 2, 2, 2)(relu5_4)
	

	Vgg_model = nn.gModule({input},{relu3_2})
--	Vgg_model = nn.gModule({input},{relu4_1})

--	print(Vgg_model.modules[2].weight)
--	print(torch.type(Vgg_model.modules[2]))
--	for i, node in pairs(Vgg_model.modules[2]) do
--		print(i)
--	end
	Vgg_model:cuda()

	proto_file = 'dataset/VGG_ILSVRC_19_layers_deploy.prototxt'
	model_file = 'dataset/VGG_ILSVRC_19_layers.caffemodel'
	loadcaffe_backend = 'cudnn'
	
	local cnn = loadcaffe.load(proto_file, model_file, loadcaffe_backend):cuda()
	--print(cnn)  -- org cnn
	while #cnn > 21 do
		cnn:remove(#cnn)
	end
    --print(cnn)  -- crop cnn
	
	local num_layers = #Vgg_model.modules
	print(num_layers)
	for i = 2,num_layers do
		if torch.type(cnn:get(i-1)) == 'cudnn.SpatialConvolution' then 		
			local w = cnn:get(i-1).weight:clone()
			local b = cnn:get(i-1).bias:clone()
			Vgg_model.modules[i].weight = w:cuda()
			Vgg_model.modules[i].bias   = b:cuda()
			--print(torch.all(torch.eq(cnn:get(i-1).weight, Vgg_model.modules[i].weight)))
		end
	end

--[[
	Vgg_model:zeroGradParameters()
	cnn:zeroGradParameters()
	Vgg_model:evaluate()
	cnn:evaluate()
	
	a = torch.CudaTensor(1,3,32,32):fill(0)
	t = torch.CudaTensor(1,512,4,4):fill(1)
--	a = preprocess_image(a)
	cri1 = nn.MSECriterion():cuda()
	b1 = Vgg_model:forward(a)
	c1 = cri1:forward(b1,t)	
	d1 = cri1:backward(b1,t)
	e1 = Vgg_model:backward(a,d1)
	
	cri2 = nn.MSECriterion():cuda()
	b2 = cnn:forward(a)
	c2 = cri2:forward(b2,t)
	d2 = cri2:backward(b2,t)
	e2 = cnn:backward(a,d2)
	
	print(b1,b2)
	print(#e1)
	if torch.all(torch.eq(e1, e2)) then
		print('they are the same')
	else
		print(torch.sum(e1-e2))
		--print(e1[1][2][2],e2[1][2][2])
	end
--]]		
		
	return Vgg_model
end

--Vgg_model = createVggmodel()

--a = torch.CudaTensor(1,3,32,32):fill(0)
--b1 = Vgg_model:forward(a)
--print(b1[4][{{1},{512}}])

function createVggmodel_trim()
	input = nn.Identity()()
	conv1_1 = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1)(input)   -- 2
	relu1_1 = cudnn.ReLU(true)(conv1_1)
	conv1_2 = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1)(relu1_1)
	relu1_2 = cudnn.ReLU(true)(conv1_2)
	pool1   = cudnn.SpatialMaxPooling(2, 2, 2, 2)(relu1_2)
	
	conv2_1 = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1)(pool1)  -- 7
	relu2_1 = cudnn.ReLU(true)(conv2_1)
	conv2_2 = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1)(relu2_1)
	relu2_2 = cudnn.ReLU(true)(conv2_2)
	pool2   = cudnn.SpatialMaxPooling(2, 2, 2, 2)(relu2_2)
	
	conv3_1 = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1)(pool2)  -- 12
	relu3_1 = cudnn.ReLU(true)(conv3_1)
	conv3_2 = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1)(relu3_1)
	relu3_2 = cudnn.ReLU(true)(conv3_2)
--	conv3_3 = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)(relu3_2)
--	relu3_3 = cudnn.ReLU(true)(conv3_3)
--	conv3_4 = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)(relu3_3)
--	relu3_4 = cudnn.ReLU(true)(conv3_4)
--	pool3   = cudnn.SpatialMaxPooling(2, 2, 2, 2)(relu3_4)
	
--	conv4_1 = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)(pool3)  -- 21
--	relu4_1 = cudnn.ReLU(true)(conv4_1) 
--	conv4_2 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)(relu4_1)
--	relu4_2 = cudnn.ReLU(true)(conv4_2)
--	conv4_3 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)(relu4_2)
--	relu4_3 = cudnn.ReLU(true)(conv4_3)
--	conv4_4 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)(relu4_3)
--	relu4_4 = cudnn.ReLU(true)(conv4_4)
--	pool4   = cudnn.SpatialMaxPooling(2, 2, 2, 2)(relu4_4)
	
--	conv5_1 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)(pool4)   -- 30
--	relu5_1 = cudnn.ReLU(true)(conv5_1)
--	conv5_2 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)(relu5_1)
--	relu5_2 = cudnn.ReLU(true)(conv5_2)
--	conv5_3 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)(relu5_2)
--	relu5_3 = cudnn.ReLU(true)(conv5_3)
--	conv5_4 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)(relu5_3)
--	relu5_4 = cudnn.ReLU(true)(conv5_4)
--	pool5   = cudnn.SpatialMaxPooling(2, 2, 2, 2)(relu5_4)
	

	Vgg_model = nn.gModule({input},{relu3_2})
--	Vgg_model = nn.gModule({input},{relu4_1})

--	print(Vgg_model.modules[2].weight)
--	print(torch.type(Vgg_model.modules[2]))
--	for i, node in pairs(Vgg_model.modules[2]) do
--		print(i)
--	end
	Vgg_model:cuda()

	proto_file = 'dataset/VGG_ILSVRC_19_layers_deploy.prototxt'
	model_file = 'dataset/VGG_ILSVRC_19_layers.caffemodel'
	loadcaffe_backend = 'cudnn'
	
	local cnn = loadcaffe.load(proto_file, model_file, loadcaffe_backend):cuda()
	--print(cnn)  -- org cnn
	while #cnn > 21 do
		cnn:remove(#cnn)
	end
    --print(cnn)  -- crop cnn
	
	local num_layers = #Vgg_model.modules
	print(num_layers)
	for i = 2,num_layers do
		if torch.type(cnn:get(i-1)) == 'cudnn.SpatialConvolution' then 		
			local w = cnn:get(i-1).weight:clone()
			local b = cnn:get(i-1).bias:clone()
			Vgg_model.modules[i].weight = w:cuda()
			Vgg_model.modules[i].bias   = b:cuda()
			--print(torch.all(torch.eq(cnn:get(i-1).weight, Vgg_model.modules[i].weight)))
		end
	end
		
	return Vgg_model
end