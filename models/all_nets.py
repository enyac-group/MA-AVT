import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
# from ipdb import set_trace
import timm
from torch import Tensor
from typing import Optional, Any
from einops import rearrange, repeat
import types
from timm.models.vision_transformer import Attention
import timm
# import loralib as lora
from .my_layers import PHMLinear
# from transformers.activations import get_activation
import torch.distributed as dist

from models.my_vit import my_vit
import torch.distributed as dist


def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def attn_new_forward(self, x, return_attn=False):
	B, N, C = x.shape
	qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
	q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

	attn = (q @ k.transpose(-2, -1)) * self.scale
	orig_attn = attn
	attn = attn.softmax(dim=-1)
	attn = self.attn_drop(attn)

	x = (attn @ v).transpose(1, 2).reshape(B, N, C)
	x = self.proj(x)
	x = self.proj_drop(x)

	if return_attn:
		return x, orig_attn

	return x


class VisualAdapter(nn.Module):
	"""Conventional Adapter layer, in which the weights of up and down sampler modules
	are parameters and are optimized."""

	def __init__(self, input_dim, output_dim, adapter_kind, dim_list=None, layer_idx=0, reduction_factor=16, opt=None ,use_bn=True, use_gate=True):
		super().__init__()
		self.adapter_kind = adapter_kind
		self.use_bn = use_bn
		self.is_multimodal = opt.is_multimodal
		self.opt = opt

		if use_gate:
			self.gate = nn.Parameter(torch.zeros(1))
		else:
			self.gate = None

		if adapter_kind == "bottleneck" and self.is_multimodal:
			self.down_sample_size = input_dim // reduction_factor
			### -----> attetnion
			self.my_tokens = nn.Parameter(torch.rand((self.opt.num_tokens, input_dim)))

			self.gate_av = nn.Parameter(torch.zeros(1))
			self.gate_tk = nn.Parameter(torch.ones(1))

			### <------

			self.activation = nn.ReLU(inplace=True)

			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)

			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)

			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)

			### -------> yb: add
			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
			### <---------

		elif adapter_kind == "bottleneck":
			self.down_sample_size = input_dim // reduction_factor
			self.activation = nn.ReLU(inplace=True)


			self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)
			# nn.init.zeros_(self.down_sampler) # yb:for lora

			self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt.num_conv_group, bias=False)

			if use_bn:
				self.bn1 = nn.BatchNorm2d(self.down_sample_size)
				self.bn2 = nn.BatchNorm2d(output_dim)

			### -------> yb: add
			if self.opt.is_before_layernorm:
				self.ln_before = nn.LayerNorm(output_dim)
			if self.opt.is_post_layernorm:
				self.ln_post = nn.LayerNorm(output_dim)
			### <---------

		elif adapter_kind == "basic":
			self.activation = nn.ReLU(inplace=True)
			# self.conv = nn.Conv2d(input_dim, output_dim, 1, bias=False)
			self.conv = nn.Linear(input_dim, output_dim, bias=False)

			if use_bn:
				self.bn = nn.BatchNorm1d(output_dim)

		else:
			raise NotImplementedError

	def forward(self, x, vis_token=None):
		if self.adapter_kind == "bottleneck" and self.is_multimodal:

			### -------> high dim att
			rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))
			att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))

			att_v2tk = F.softmax(att_v2tk, dim=-1)
			rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0,2,1))

			rep_token = rep_token + self.gate_tk*rep_token_res

			att_tk2x = torch.bmm(x.squeeze(-1).permute(0,2,1), rep_token.permute(0,2,1))

			att_tk2x = F.softmax(att_tk2x, dim=-1)
			x_res = torch.bmm(att_tk2x, rep_token).permute(0,2,1).unsqueeze(-1)

			x = x + self.gate_av*x_res.contiguous()
			### <----------
			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

			z = self.down_sampler(x)

			## <----

			if self.use_bn:
				z = self.bn1(z)

			z = self.activation(z)
			output = self.up_sampler(z)

			if self.use_bn:
				output = self.bn2(output)

		elif self.adapter_kind == "bottleneck":

			if self.opt.is_before_layernorm:
				x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

			z = self.down_sampler(x)

			if self.use_bn:
				z = self.bn1(z)

			output = self.up_sampler(z)
			if self.use_bn:
				output = self.bn2(output)

		elif self.adapter_kind == "basic":
			output = self.conv(x)
			if self.use_bn:
				output = self.bn(rearrange(output, 'N C L -> N L C') )
				output = rearrange(output, 'N L C -> N C L')

		if self.opt.is_post_layernorm:
			output = self.ln_post(output.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

		if self.gate is not None:
			output = self.gate * output

		return output


class LAVISH(nn.Module):
	def __init__(self, args):
		super(LAVISH, self).__init__()

		self.opt = args

		if args.vis_encoder_type == 'vit':

			if args.vit_type == "tiny":
				self.ViT = my_vit(name='vit_tiny_patch16_224_in21k')
			elif args.vit_type == "small":
				self.ViT = my_vit(name='vit_small_patch16_224_in21k')
			elif args.vit_type == "base":
				self.ViT = my_vit(name='vit_base_patch16_224_in21k')
			elif args.vit_type == "large":
				self.ViT = my_vit(name='vit_large_patch16_224_in21k')
			else:
				print("Nothing found. TinyViT is loading")
				self.ViT = my_vit(name='vit_tiny_patch16_224_in21k')

			dim = self.ViT.v.patch_embed.proj.out_channels
			self.mlp_class = nn.Linear(dim*2, 512) 
			self.mlp_class_2 = nn.Linear(512, args.num_class)

		self.total_layers = len(self.ViT.v.blocks)

		hidden_list = []

		if args.vis_encoder_type == 'vit':
			for idx_layer, my_blk in enumerate(self.ViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)

				if idx_layer == (self.total_layers - 1):
					my_blk.attn.forward = types.MethodType(attn_new_forward, my_blk.attn)


		# if self.opt.is_audio_adapter_p1:
		self.audio_adapter_blocks_p1 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=args, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
			for i in range(len(hidden_list))])

		self.vis_adapter_blocks_p1 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=args, use_bn=self.opt.is_bn, use_gate=True)
			for i in range(len(hidden_list))])

		self.audio_adapter_blocks_p2 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=args, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
			for i in range(len(hidden_list))])

		self.vis_adapter_blocks_p2 = nn.ModuleList([
			VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=args, use_bn=self.opt.is_bn, use_gate=True)
			for i in range(len(hidden_list))])


	def forward_vit(self, audio, vis, target, return_attn=False):
		b, t, c, w, h = vis.shape

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)

		f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True)
		f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True)

		bs, dim, wa, ha = patch_info_audio
		bs, dim, wv, wv = patch_info_vis

		layer_count = 0

		for idx_layer, blk in enumerate(self.ViT.v.blocks) :
			if idx_layer >= self.opt.start_tune_layers:
				f_a_res = self.audio_adapter_blocks_p1[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

				if (idx_layer == self.total_layers - 1) and return_attn:
					x, attn = blk.attn(blk.norm1(f_v), return_attn=True)
					f_v = f_v + blk.drop_path1(blk.ls1(x))
				else:
					f_v = f_v + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f_v))))

				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f_a))))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)

				f_a_res = self.audio_adapter_blocks_p2[idx_layer](f_a.permute(0,2,1).unsqueeze(-1), f_v.permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[idx_layer](f_v.permute(0,2,1).unsqueeze(-1), f_a.permute(0,2,1).unsqueeze(-1))

				f_v = f_v + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_v))))
				f_v = f_v + f_v_res.squeeze(-1).permute(0,2,1)

				f_a = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_a))))
				f_a = f_a + f_a_res.squeeze(-1).permute(0,2,1)

			layer_count += 1

		f_v = self.ViT.v.norm(f_v)
		f_a = self.ViT.v.norm(f_a)

		v_cls = f_v[:, 0:1].clone()
		a_cls = f_a[:, 0:1].clone()

		out_av = torch.cat((a_cls, v_cls), dim=-1)

		out_av = rearrange(out_av, 'b t p -> (b t) p')

		p_av = self.mlp_class(out_av)

		p_av = self.mlp_class_2(p_av)

		if return_attn:
			return p_av, attn

		return p_av
			
	def forward(self, audio, vis, target, return_attn=False):
		if self.opt.vis_encoder_type == 'swin':
			return self.forward_swin(audio, vis)
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, target, return_attn)


class MBT(nn.Module):
	def __init__(self, args):
		super(MBT, self).__init__()
		assert args.vis_encoder_type == 'vit', "only vit is supported for now"

		self.opt = args

		if args.vis_encoder_type == 'vit':
			if args.vit_type == "tiny":
				self.AViT = my_vit('vit_tiny_patch16_224_in21k', pretrained=args.pretrained)
				self.IViT = my_vit('vit_tiny_patch16_224_in21k', pretrained=args.pretrained)

			elif args.vit_type == "small":
				self.AViT = my_vit(name='vit_small_patch16_224_in21k', pretrained=args.pretrained)
				self.IViT = my_vit(name='vit_small_patch16_224_in21k', pretrained=args.pretrained)

			elif args.vit_type == "base":
				self.AViT = my_vit(name='vit_base_patch16_224_in21k', pretrained=args.pretrained)
				self.IViT = my_vit(name='vit_base_patch16_224_in21k', pretrained=args.pretrained)

			elif args.vit_type == "large":
				self.AViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=args.pretrained)
				self.IViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=args.pretrained)

			else:
				print("Nothing found. TinyViT is loading")
				self.AViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=args.pretrained)
				self.IViT =  my_vit(name='vit_large_patch16_224_in21k', pretrained=args.pretrained)

			self.dim = self.AViT.v.patch_embed.proj.out_channels
			self.mlp_class = nn.Linear(self.dim*2, 512)
			self.mlp_class_2 = nn.Linear(512, args.num_class)
		
		hidden_list = []
		down_in_dim = []
		down_out_dim = []

		self.total_layers = len(self.AViT.v.blocks)

		if args.vis_encoder_type == 'vit':
			for idx_layer, my_blk in enumerate(self.AViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)

				if idx_layer == (self.total_layers - 1):
					my_blk.attn.forward = types.MethodType(attn_new_forward, my_blk.attn)

			self.shared_tokens = nn.Parameter(torch.randn(5, hidden_d_size))

	def forward_vit(self, audio, vis, target, return_attn=True):
		b, t, c, w, h = vis.shape

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)

		f_a, patch_info_audio = self.AViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True)
		f_v, patch_info_vis = self.IViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True)

		bs, dim, wa, ha = patch_info_audio
		bs, dim, wv, wv = patch_info_vis

		layer_count = 0
		shared_tokens = repeat(self.shared_tokens, 'len dim -> b len dim', b=f_v.shape[0])

		f_a = torch.cat([
			f_a,
			shared_tokens
		], dim=1)

		f_v = torch.cat([
			f_v,
			shared_tokens
		], dim=1)


		for i, (ablk, iblk) in enumerate(zip(self.AViT.v.blocks, self.IViT.v.blocks)):
			if (i == self.total_layers - 1) and return_attn:
				x, attn = iblk.attn(iblk.norm1(f_v), return_attn=True)
				f_v = f_v + iblk.drop_path1(iblk.ls1(x))
			else:
				f_v = f_v + iblk.drop_path1(iblk.ls1(iblk.attn(iblk.norm1(f_v))))
			
			f_v = f_v + iblk.drop_path2(iblk.ls2(iblk.mlp(iblk.norm2(f_v))))

			f_a = f_a + ablk.drop_path1(ablk.ls1(ablk.attn(ablk.norm1(f_a))))
			f_a = f_a + ablk.drop_path2(ablk.ls2(ablk.mlp(ablk.norm2(f_a))))

			layer_count += 1


		f_v = self.IViT.v.norm(f_v)
		f_a = self.AViT.v.norm(f_a)

		v_cls = f_v[:, 0:1].squeeze().clone()
		a_cls = f_a[:, 0:1].squeeze().clone()

		v_cls = v_cls.view(b, t, self.dim).mean(dim=1)
		a_cls = a_cls.view(b, t, self.dim).mean(dim=1)

		out_av = torch.cat((a_cls, v_cls), dim=-1)

		p_av = self.mlp_class(out_av)

		# f_a = rearrange(audio, 'b t dim -> (b t) dim')
		p_av = self.mlp_class_2(p_av)

		if return_attn:
			return p_av, attn

		return p_av

	def forward(self, audio, vis, target, return_attn=False):
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, target, return_attn)



class Shared_Transformer(nn.Module):
	def __init__(self, args):
		super(Shared_Transformer, self).__init__()
		assert args.vis_encoder_type == 'vit', "only vit is supported for now"

		self.opt = args

		if args.vis_encoder_type == 'vit':
			if args.vit_type == "tiny":
				self.ViT = my_vit('vit_tiny_patch16_224_in21k', pretrained=args.pretrained)
			elif args.vit_type == "small":
				self.ViT = my_vit(name='vit_small_patch16_224_in21k', pretrained=args.pretrained)
			elif args.vit_type == "base":
				self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrained=args.pretrained)
			elif args.vit_type == "large":
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=args.pretrained)
			else:
				print("Nothing found. TinyViT is loading")
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=args.pretrained)

			self.dim = self.ViT.v.patch_embed.proj.out_channels
			self.mlp_class = nn.Linear(self.dim*2, 512)
			self.mlp_class_2 = nn.Linear(512, args.num_class)

		hidden_list = []
		down_in_dim = []
		down_out_dim = []

		self.total_layers = len(self.ViT.v.blocks)

		if args.vis_encoder_type == 'vit':
			for idx_layer, my_blk in enumerate(self.ViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)

				if idx_layer == (self.total_layers - 1):
					my_blk.attn.forward = types.MethodType(attn_new_forward, my_blk.attn)


	def forward_vit(self, audio, vis, target, return_attn=True):
		b, t, c, w, h = vis.shape

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)

		f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True)
		f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True)

		bs, dim, wa, ha = patch_info_audio
		bs, dim, wv, wv = patch_info_vis

		layer_count = 0

		for i, blk in enumerate(self.ViT.v.blocks):
			if (i == self.total_layers - 1) and return_attn:
				x, attn = blk.attn(blk.norm1(f_v), return_attn=True)
				f_v = f_v + blk.drop_path1(blk.ls1(x))
			else:
				f_v = f_v + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f_v))))
			
			f_v = f_v + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_v))))

			f_a = f_a + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f_a))))
			f_a = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_a))))

			layer_count += 1


		f_v = self.ViT.v.norm(f_v)
		f_a = self.ViT.v.norm(f_a)

		v_cls = f_v[:,0:1].squeeze().clone()
		a_cls = f_a[:,0:1].squeeze().clone()

		v_cls = v_cls.view(b, t, self.dim).mean(dim=1)
		a_cls = a_cls.view(b, t, self.dim).mean(dim=1)

		out_av = torch.cat((a_cls, v_cls), dim=-1)

		p_av = self.mlp_class(out_av)

		# f_a = rearrange(audio, 'b t dim -> (b t) dim')
		p_av = self.mlp_class_2(p_av)

		if return_attn:
			return p_av, attn

		return p_av

	def forward(self, audio, vis, target, return_attn=False):
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, target, return_attn)


class Image_Transformer(nn.Module):
	def __init__(self, args):
		super(Image_Transformer, self).__init__()
		assert args.vis_encoder_type == 'vit', "only vit is supported for now"

		self.opt = args

		if args.vis_encoder_type == 'vit':
			if args.vit_type == "tiny":
				self.ViT = my_vit('vit_tiny_patch16_224_in21k', pretrained=args.pretrained)
			elif args.vit_type == "small":
				self.ViT = my_vit(name='vit_small_patch16_224_in21k', pretrained=args.pretrained)
			elif args.vit_type == "base":
				self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrained=args.pretrained)
			elif args.vit_type == "large":
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=args.pretrained)
			else:
				print("Nothing found. TinyViT is loading")
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=args.pretrained)

			self.dim = self.ViT.v.patch_embed.proj.out_channels
			self.mlp_class = nn.Linear(self.dim, 512)
			self.mlp_class_2 = nn.Linear(512, args.num_class)

		hidden_list = []
		down_in_dim = []
		down_out_dim = []

		self.total_layers = len(self.ViT.v.blocks)

		if args.vis_encoder_type == 'vit':
			for idx_layer, my_blk in enumerate(self.ViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)

				if idx_layer == (self.total_layers - 1):
					my_blk.attn.forward = types.MethodType(attn_new_forward, my_blk.attn)


	def forward_vit(self, audio, vis, target, return_attn=True):
		b, t, c, w, h = vis.shape

		f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True)

		bs, dim, wv, wv = patch_info_vis

		layer_count = 0

		for i, blk in enumerate(self.ViT.v.blocks):
			if (i == self.total_layers - 1) and return_attn:
				x, attn = blk.attn(blk.norm1(f_v), return_attn=True)
				f_v = f_v + blk.drop_path1(blk.ls1(x))
			else:
				f_v = f_v + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f_v))))
			
			f_v = f_v + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_v))))


			layer_count += 1


		f_v = self.ViT.v.norm(f_v)
		v_cls = f_v[:,0:1].squeeze().clone()
		v_cls = v_cls.view(b, t, self.dim).mean(dim=1)
		out_av = v_cls
		p_av = self.mlp_class(out_av)
		p_av = self.mlp_class_2(p_av)

		if return_attn:
			return p_av, attn

		return p_av

	def forward(self, audio, vis, target, return_attn=False):
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, target, return_attn)


class Audio_Transformer(nn.Module):
	def __init__(self, args):
		super(Audio_Transformer, self).__init__()
		assert args.vis_encoder_type == 'vit', "only vit is supported for now"

		self.opt = args

		if args.vis_encoder_type == 'vit':
			if args.vit_type == "tiny":
				self.ViT = my_vit('vit_tiny_patch16_224_in21k', pretrained=args.pretrained)
			elif args.vit_type == "small":
				self.ViT = my_vit(name='vit_small_patch16_224_in21k', pretrained=args.pretrained)
			elif args.vit_type == "base":
				self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrained=args.pretrained)
			elif args.vit_type == "large":
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=args.pretrained)
			else:
				print("Nothing found. TinyViT is loading")
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=args.pretrained)

			self.dim = self.ViT.v.patch_embed.proj.out_channels
			self.mlp_class = nn.Linear(self.dim, 512)
			self.mlp_class_2 = nn.Linear(512, args.num_class)

		hidden_list = []
		down_in_dim = []
		down_out_dim = []

		self.total_layers = len(self.ViT.v.blocks)

		if args.vis_encoder_type == 'vit':
			for idx_layer, my_blk in enumerate(self.ViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)

				if idx_layer == (self.total_layers - 1):
					my_blk.attn.forward = types.MethodType(attn_new_forward, my_blk.attn)


	def forward_vit(self, audio, vis, target, return_attn=True):
		b, t, c, w, h = vis.shape

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)

		f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True)

		bs, dim, wa, ha = patch_info_audio

		layer_count = 0

		for i, blk in enumerate(self.ViT.v.blocks):
			f_a = f_a + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(f_a))))
			f_a = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_a))))

			layer_count += 1

		f_a = self.ViT.v.norm(f_a)

		a_cls = f_a[:, 0:1].squeeze().clone()
		a_cls = a_cls.view(b, t, self.dim).mean(dim=1)

		out_av = a_cls

		p_av = self.mlp_class(out_av)
		p_av = self.mlp_class_2(p_av)

		if return_attn:
			return p_av, attn

		return p_av

	def forward(self, audio, vis, target, return_attn=False):
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, target, return_attn)



class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x



class MA_AVT(nn.Module):
	def __init__(self, opt):
		super(MA_AVT, self).__init__()
		
		assert opt.vis_encoder_type == 'vit', "only vit is supported for now"

		self.opt = opt
		self.ngpu = opt.ngpu
		self.rank = opt.rank

		if opt.vis_encoder_type == 'vit':
			if opt.vit_type == "tiny":
				self.ViT = my_vit('vit_tiny_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "small":
				self.ViT = my_vit(name='vit_small_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "base":
				self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "large":
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)
			else:
				print("Nothing found. TinyViT is loading")
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)

			self.dim = self.ViT.v.patch_embed.proj.out_channels

			self.fg_class = nn.Linear(self.dim * 2, opt.num_class)

		hidden_list = []
		self.total_layers = len(self.ViT.v.blocks)

		self.contrastive = opt.contrastive
		self.LSA = opt.LSA

		if opt.contrastive in ['final' or 'blockwise_cmn']:
			self.audio_proj = nn.Linear(self.dim, 512)
			self.vis_proj = nn.Linear(self.dim, 512)
			self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

		elif opt.contrastive == 'blockwise_sep':
			self.audio_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 

			self.vis_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 
			
			self.logit_scale = nn.ParameterList([
				nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
			for i in range(self.total_layers)])

		if opt.vis_encoder_type == 'vit':
			for idx_layer, my_blk in enumerate(self.ViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)

				my_blk.attn.forward = types.MethodType(attn_new_forward, my_blk.attn)

		self.lavish_adapter = opt.lavish_adapter

		if opt.lavish_adapter:
			self.audio_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

			self.audio_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

		self.unimodal_token = opt.unimodal_token
		self.multimodal_token = opt.multimodal_token
		self.layerwise_token = opt.layerwise_token # "None" or "common" or "unique"
		self.layerwise_attn = opt.layerwise_attn # "None" or "common" or "unique"

		if opt.unimodal_token:
			self.audio_tokens = nn.Parameter(torch.randn(opt.n_audio_tokens, hidden_d_size))
			self.visual_tokens = nn.Parameter(torch.randn(opt.n_vis_tokens, hidden_d_size))
		if opt.multimodal_token:
			self.shared_tokens = nn.Parameter(torch.randn(opt.n_shared_tokens, hidden_d_size))

		self.n_audio_tokens = opt.n_audio_tokens
		self.n_vis_tokens = opt.n_vis_tokens
		self.n_shared_tokens = opt.n_shared_tokens

		self.audio_attn = nn.Identity()
		self.visual_attn = nn.Identity()
		self.shared_attn = nn.Identity()

		if opt.unimodal_token and opt.LSA:
			self.audio_attn = SelfAttention(hidden_d_size)
			self.visual_attn = SelfAttention(hidden_d_size)
		if opt.multimodal_token and opt.LSA:
			self.shared_attn = SelfAttention(hidden_d_size)
		
		self.is_bg_cls = opt.bg_cls
		if opt.bg_cls:
			self.bg_cls_token = nn.Parameter(torch.randn(1, hidden_d_size))
			self.bg_class = nn.Linear(self.dim * 2, 1)
			self.bg_label = opt.bg_label

		self.grad_mod = opt.grad_mod

	def calc_cnt_loss(self, image_feat, audio_feat, bg_label, layer_count):
		gather_image_feat = [torch.zeros_like(image_feat) for _ in range(self.ngpu)]
		gather_audio_feat = [torch.zeros_like(audio_feat) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			gather_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(gather_image_feat, image_feat)
		dist.all_gather(gather_audio_feat, audio_feat)
		if self.is_bg_cls:
			dist.all_gather(gather_bg_label, bg_label)
		
		gather_image_feat[self.rank] = image_feat
		gather_audio_feat[self.rank] = audio_feat

		all_image_feat = torch.cat(gather_image_feat, dim=0)
		all_audio_feat = torch.cat(gather_audio_feat, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(gather_bg_label, dim=0)

		if self.is_bg_cls:
			mask = torch.where(all_bg_label == 0)[0]
			all_image_feat = all_image_feat[mask]
			all_audio_feat = all_audio_feat[mask]
		
		label = torch.arange(len(all_image_feat)).to(image_feat.device)
		
		if self.contrastive in ['blockwise_sep', 'blockwise_cmn']:
			pred_img = (1/0.07) * (all_image_feat @ all_audio_feat.T)	
			pred_aud = pred_img.T
		else:
			pred_img = (1/0.07)* (all_image_feat @ all_audio_feat.T)	
			pred_aud = pred_img.T

		loss = (F.cross_entropy(pred_img, label) + F.cross_entropy(pred_aud, label))/2

		return loss

	def calc_grad_coeff(self, v_cls, a_cls, label, bg_label):
		'''
		return coeff_v, coeff_a
		'''
		weight_size = self.fg_class.weight.size(1)
		out_v = (torch.mm(v_cls, torch.transpose(self.fg_class.weight[:, weight_size // 2:], 0, 1))
					+ self.fg_class.bias / 2)

		out_a = (torch.mm(a_cls, torch.transpose(self.fg_class.weight[:, :weight_size // 2], 0, 1))
					+ self.fg_class.bias / 2)

		# Modulation starts here !
		all_out_v = [torch.zeros_like(out_v) for _ in range(self.ngpu)]
		all_out_a = [torch.zeros_like(out_a) for _ in range(self.ngpu)]
		all_label = [torch.zeros_like(label) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			all_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(all_out_v, out_v)
		dist.all_gather(all_out_a, out_a)
		dist.all_gather(all_label, label)
		if self.is_bg_cls:
			dist.all_gather(all_bg_label, bg_label)

		all_out_v = torch.cat(all_out_v, dim=0)
		all_out_a = torch.cat(all_out_a, dim=0)
		all_label = torch.cat(all_label, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(all_bg_label, dim=0)

			mask = torch.where(all_bg_label == 0)[0]
			all_out_v = all_out_v[mask]
			all_out_a = all_out_a[mask]
			all_label = all_label[mask]

		score_v = sum([F.softmax(all_out_v)[i][all_label[i]] for i in range(all_out_v.size(0))])
		score_a = sum([F.softmax(all_out_a)[i][all_label[i]] for i in range(all_out_a.size(0))])

		ratio_v = score_v / score_a
		ratio_a = 1 / ratio_v

		if ratio_v > 1:
			coeff_v = 1 - F.tanh(0.1 * F.relu(ratio_v))
			coeff_a = 1
		else:
			coeff_a = 1 - F.tanh(0.1 * F.relu(ratio_a))
			coeff_v = 1

		return coeff_v, coeff_a


	def calc_fg_loss(self, p_fg, label, bg_label):
		if self.is_bg_cls:
			mask = torch.where(bg_label == 0)[0]

			label = label[mask]
			p_fg = p_fg[mask]

		loss = F.cross_entropy(p_fg, label)

		return loss

	def calc_bg_loss(self, p_bg, bg_label):
		loss = F.binary_cross_entropy_with_logits(p_bg, bg_label.float())
		return loss 

	def forward_vit(self, audio, vis, label, return_attn=True):
		b, t, c, w, h = vis.shape

		label = label.view(-1)

		if self.is_bg_cls:
			bg_label = (label == self.bg_label).long()
		else:
			bg_label = None

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)

		if self.is_bg_cls:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token)
		else:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True)

		bs, dim, wa, ha = patch_info_audio
		bs, dim, wv, wv = patch_info_vis

		layer_count = 0
		cnt_loss = 0
		fg_loss = 0
		bg_loss = 0

		for i, blk in enumerate(self.ViT.v.blocks):
			if i == 0 or self.layerwise_token in ["unique", "common"]:
				if self.unimodal_token:
					audio_tokens = repeat(self.audio_tokens, 'len dim -> b len dim', b=f_a.shape[0])
					visual_tokens = repeat(self.visual_tokens, 'len dim -> b len dim', b=f_v.shape[0])

					audio_tokens = self.audio_attn(audio_tokens)
					visual_tokens = self.visual_attn(visual_tokens)

					f_a = torch.cat([
						f_a,
						audio_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						visual_tokens
					], dim=1)

				if self.multimodal_token:
					shared_tokens = repeat(self.shared_tokens, 'len dim -> b len dim', b=f_v.shape[0])
					shared_tokens = self.shared_attn(shared_tokens)
			
					f_a = torch.cat([
						f_a,
						shared_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						shared_tokens
					], dim=1)

			if self.lavish_adapter:
				f_a_res = self.audio_adapter_blocks_p1[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))

			x, attn = blk.attn(blk.norm1(f_v), return_attn=True)
			
			if self.is_bg_cls:
				vis_attn = attn[:, :, 0, 1:-(1+self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)
			else:
				vis_attn = attn[:, :, 0, 1:-(self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)

			f_v = f_v + blk.drop_path1(blk.ls1(x))
		
			x, attn = blk.attn(blk.norm1(f_a), return_attn=True)
			
			if self.is_bg_cls:
				aud_attn = attn[:, :, 0, 1:-(1+self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)
			else:
				aud_attn = attn[:, :, 0, 1:-(self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)

			f_a = f_a + blk.drop_path1(blk.ls1(x))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)
			
				f_a_res = self.audio_adapter_blocks_p2[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
	
			f_v = f_v + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_v))))
			f_a = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_a))))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)

			vis_shared_tokens =  f_v[:, -self.n_shared_tokens:, :]
			aud_shared_tokens = f_a[:, -self.n_shared_tokens:, :]

			if self.contrastive == 'blockwise_sep':
				audio_feat = F.normalize(self.audio_proj[layer_count](aud_shared_tokens.mean(dim=1)).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj[layer_count](vis_shared_tokens.mean(dim=1)).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'blockwise_cmn':
				audio_feat = F.normalize(self.audio_proj(aud_shared_tokens.mean(dim=1)).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj(vis_shared_tokens.mean(dim=1)).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'final' and i == (self.total_layers-1):
				audio_feat = F.normalize(self.audio_proj(aud_shared_tokens.mean(dim=1)).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj(vis_shared_tokens.mean(dim=1)).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			if self.layerwise_token is not None and i != (self.total_layers-1):
				f_a = f_a[:, :-(self.n_vis_tokens+self.n_shared_tokens) , :]
				f_v = f_v[:, :-(self.n_audio_tokens+self.n_shared_tokens), :]

			layer_count += 1

		if self.contrastive in ["blockwise_sep", "blockwise_cmn"]:
			cnt_loss = cnt_loss/self.total_layers

		f_v = self.ViT.v.norm(f_v)
		f_a = self.ViT.v.norm(f_a)

		v_cls = f_v[:, 0:1].squeeze()
		a_cls = f_a[:, 0:1].squeeze()

		fg_feat = torch.cat((v_cls, a_cls), dim=-1)
		p_fg = self.fg_class(fg_feat)

		if self.is_bg_cls:
			v_bg_cls = f_v[:, -(1+self.n_vis_tokens+self.n_shared_tokens):-(self.n_vis_tokens+self.n_shared_tokens)].squeeze()
			a_bg_cls = f_a[:, -(1+self.n_audio_tokens+self.n_shared_tokens):-(self.n_audio_tokens+self.n_shared_tokens)].squeeze()
			bg_feat = torch.cat((v_bg_cls, a_bg_cls), dim=-1)
			p_bg = self.bg_class(bg_feat).squeeze()
			
			bg_loss = self.calc_bg_loss(p_bg, bg_label)

		fg_loss = self.calc_fg_loss(p_fg, label, bg_label)

		loss = fg_loss + bg_loss + cnt_loss

		if self.grad_mod:
			coeff_v, coeff_a = self.calc_grad_coeff(v_cls, a_cls, label, bg_label)
		else:
			coeff_v, coeff_a = None, None

		return {
			'loss': loss,
			'fg_loss': fg_loss,
			'bg_loss': bg_loss if self.is_bg_cls else None,
			'cnt_loss': cnt_loss,
			'p_fg': p_fg,
			'p_bg': p_bg if self.is_bg_cls else None,
			'vis_attn': vis_attn,
			'coeff_a': coeff_a,
			'coeff_v': coeff_v
		}

	def forward(self, audio, vis, return_attn=False):
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, return_attn)


class AVM_VIT_RAW(nn.Module):
	def __init__(self, opt):
		super(AVM_VIT_RAW, self).__init__()
		
		assert opt.vis_encoder_type == 'vit', "only vit is supported for now"

		self.opt = opt
		self.ngpu = opt.ngpu
		self.rank = opt.rank

		if opt.vis_encoder_type == 'vit':
			if opt.vit_type == "tiny":
				self.ViT = my_vit('vit_tiny_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "small":
				self.ViT = my_vit(name='vit_small_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "base":
				self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "large":
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)
			else:
				print("Nothing found. TinyViT is loading")
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)

			self.dim = self.ViT.v.patch_embed.proj.out_channels

			self.fg_class = nn.Linear(self.dim * 2, opt.num_class)

		hidden_list = []
		self.total_layers = len(self.ViT.v.blocks)

		self.contrastive = opt.contrastive

		if opt.contrastive in ['final' or 'blockwise_cmn']:
			self.audio_proj = nn.Linear(self.dim, 512)
			self.vis_proj = nn.Linear(self.dim, 512)
			self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

		elif opt.contrastive == 'blockwise_sep':
			self.audio_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 

			self.vis_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 
			
			self.logit_scale = nn.ParameterList([
				nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
			for i in range(self.total_layers)])

		if opt.vis_encoder_type == 'vit':
			for idx_layer, my_blk in enumerate(self.ViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)

				my_blk.attn.forward = types.MethodType(attn_new_forward, my_blk.attn)

		self.lavish_adapter = opt.lavish_adapter

		if opt.lavish_adapter:
			self.audio_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

			self.audio_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

		self.unimodal_token = opt.unimodal_token
		self.multimodal_token = opt.multimodal_token
		self.layerwise_token = opt.layerwise_token # "None" or "common" or "unique"
		self.layerwise_attn = opt.layerwise_attn # "None" or "common" or "unique"


		if opt.layerwise_token == "unique":
			self.audio_tokens = nn.ParameterList([
				nn.Parameter(torch.randn(opt.n_audio_tokens, hidden_d_size))
			for i in range(self.total_layers)])

			self.visual_tokens = nn.ParameterList([
				nn.Parameter(torch.randn(opt.n_vis_tokens, hidden_d_size))
			for i in range(self.total_layers)])

			if opt.multimodal_token:
				self.shared_tokens = nn.ParameterList([
					nn.Parameter(torch.randn(opt.n_shared_tokens, hidden_d_size))
				for i in range(self.total_layers)])
			
		else:
			self.audio_tokens = nn.Parameter(torch.randn(opt.n_audio_tokens, hidden_d_size))
			self.visual_tokens = nn.Parameter(torch.randn(opt.n_vis_tokens, hidden_d_size))
			if opt.multimodal_token:
				self.shared_tokens = nn.Parameter(torch.randn(opt.n_shared_tokens, hidden_d_size))

		self.n_audio_tokens = opt.n_audio_tokens
		self.n_vis_tokens = opt.n_vis_tokens
		self.n_shared_tokens = opt.n_shared_tokens

		if opt.layerwise_attn == "unique":
			self.audio_attn = nn.ModuleList([
				SelfAttention(hidden_d_size)
			for i in range(self.total_layers)])

			self.visual_attn = nn.ModuleList([
				SelfAttention(hidden_d_size)
			for i in range(self.total_layers)])

			if opt.multimodal_token:
				self.shared_attn = nn.ModuleList([
					SelfAttention(hidden_d_size)
				for i in range(self.total_layers)])
			
		else:
			self.audio_attn = SelfAttention(hidden_d_size)
			self.visual_attn = SelfAttention(hidden_d_size)
			if opt.multimodal_token:
				self.shared_attn = SelfAttention(hidden_d_size)

		self.is_bg_cls = opt.bg_cls
		if opt.bg_cls:
			self.bg_cls_token = nn.Parameter(torch.randn(1, hidden_d_size))
			# self.bg_vis_cls_token = nn.Parameter(torch.randn(1, hidden_d_size))
			self.bg_class = nn.Linear(self.dim * 2, 1)
			self.bg_label = opt.bg_label

		self.grad_mod = opt.grad_mod


	def calc_cnt_loss(self, image_feat, audio_feat, bg_label, layer_count):
		gather_image_feat = [torch.zeros_like(image_feat) for _ in range(self.ngpu)]
		gather_audio_feat = [torch.zeros_like(audio_feat) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			gather_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(gather_image_feat, image_feat)
		dist.all_gather(gather_audio_feat, audio_feat)
		if self.is_bg_cls:
			dist.all_gather(gather_bg_label, bg_label)
		
		gather_image_feat[self.rank] = image_feat
		gather_audio_feat[self.rank] = audio_feat

		all_image_feat = torch.cat(gather_image_feat, dim=0)
		all_audio_feat = torch.cat(gather_audio_feat, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(gather_bg_label, dim=0)

		if self.is_bg_cls:
			mask = torch.where(all_bg_label == 0)[0]
			all_image_feat = all_image_feat[mask]
			all_audio_feat = all_audio_feat[mask]
		
		label = torch.arange(len(all_image_feat)).to(image_feat.device)
		
		if self.contrastive in ['blockwise_sep', 'blockwise_cmn']:
			pred_img = self.logit_scale[layer_count] * (all_image_feat @ all_audio_feat.T)	
			pred_aud = pred_img.T
		else:
			pred_img = self.logit_scale * (all_image_feat @ all_audio_feat.T)	
			pred_aud = pred_img.T

		loss = (F.cross_entropy(pred_img, label) + F.cross_entropy(pred_aud, label))/2

		return loss

	def calc_grad_coeff(self, v_cls, a_cls, label, bg_label):
		'''
		return coeff_v, coeff_a
		'''
		weight_size = self.fg_class.weight.size(1)
		out_v = (torch.mm(v_cls, torch.transpose(self.fg_class.weight[:, weight_size // 2:], 0, 1))
					+ self.fg_class.bias / 2)

		out_a = (torch.mm(a_cls, torch.transpose(self.fg_class.weight[:, :weight_size // 2], 0, 1))
					+ self.fg_class.bias / 2)

		# Modulation starts here !
		all_out_v = [torch.zeros_like(out_v) for _ in range(self.ngpu)]
		all_out_a = [torch.zeros_like(out_a) for _ in range(self.ngpu)]
		all_label = [torch.zeros_like(label) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			all_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(all_out_v, out_v)
		dist.all_gather(all_out_a, out_a)
		dist.all_gather(all_label, label)
		if self.is_bg_cls:
			dist.all_gather(all_bg_label, bg_label)

		all_out_v = torch.cat(all_out_v, dim=0)
		all_out_a = torch.cat(all_out_a, dim=0)
		all_label = torch.cat(all_label, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(all_bg_label, dim=0)

			mask = torch.where(all_bg_label == 0)[0]
			all_out_v = all_out_v[mask]
			all_out_a = all_out_a[mask]
			all_label = all_label[mask]

		score_v = sum([F.softmax(all_out_v)[i][all_label[i]] for i in range(all_out_v.size(0))])
		score_a = sum([F.softmax(all_out_a)[i][all_label[i]] for i in range(all_out_a.size(0))])

		ratio_v = score_v / score_a
		ratio_a = 1 / ratio_v

		if ratio_v > 1:
			coeff_v = 1 - F.tanh(0.1 * F.relu(ratio_v))
			coeff_a = 1
		else:
			coeff_a = 1 - F.tanh(0.1 * F.relu(ratio_a))
			coeff_v = 1

		return coeff_v, coeff_a


	def calc_fg_loss(self, p_fg, label, bg_label):
		if self.is_bg_cls:
			mask = torch.where(bg_label == 0)[0]

			label = label[mask]
			p_fg = p_fg[mask]

		loss = F.cross_entropy(p_fg, label)

		return loss

	def calc_bg_loss(self, p_bg, bg_label):
		loss = F.binary_cross_entropy_with_logits(p_bg, bg_label.float())
		return loss 

	def forward_vit(self, audio, vis, label, return_attn=True):
		b, t, c, w, h = vis.shape

		label = label.view(-1)

		if self.is_bg_cls:
			bg_label = (label == self.bg_label).long()
		else:
			bg_label = None

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)

		if self.is_bg_cls:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token)
		else:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True)

		bs, dim, wa, ha = patch_info_audio
		bs, dim, wv, wv = patch_info_vis

		layer_count = 0
		cnt_loss = 0
		fg_loss = 0
		bg_loss = 0

		for i, blk in enumerate(self.ViT.v.blocks):
			if i == 0 or self.layerwise_token in ["unique", "common"]:
				if self.layerwise_token == "unique":
					audio_tokens = repeat(self.audio_tokens[layer_count], 'len dim -> b len dim', b=f_a.shape[0])
					visual_tokens = repeat(self.visual_tokens[layer_count], 'len dim -> b len dim', b=f_v.shape[0])
					if self.multimodal_token:
						shared_tokens = repeat(self.shared_tokens[layer_count], 'len dim -> b len dim', b=f_v.shape[0])
				else:
					audio_tokens = repeat(self.audio_tokens, 'len dim -> b len dim', b=f_a.shape[0])
					visual_tokens = repeat(self.visual_tokens, 'len dim -> b len dim', b=f_v.shape[0])
					if self.multimodal_token:
						shared_tokens = repeat(self.shared_tokens, 'len dim -> b len dim', b=f_v.shape[0])

				if self.layerwise_attn == "unique":
					audio_tokens = self.audio_attn[layer_count](audio_tokens)
					visual_tokens = self.visual_attn[layer_count](visual_tokens)
					if self.multimodal_token:
						shared_tokens = self.shared_attn[layer_count](shared_tokens)
				elif self.layerwise_attn == "common" or i == 0:
					audio_tokens = self.audio_attn(audio_tokens)
					visual_tokens = self.visual_attn(visual_tokens)
					if self.multimodal_token:
						shared_tokens = self.shared_attn(shared_tokens)
		
				if self.multimodal_token:
					f_a = torch.cat([
						f_a,
						audio_tokens,
						shared_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						visual_tokens,
						shared_tokens
					], dim=1)
				else:
					f_a = torch.cat([
						f_a,
						audio_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						visual_tokens
					], dim=1)

			if self.lavish_adapter:
				f_a_res = self.audio_adapter_blocks_p1[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))

			x, attn = blk.attn(blk.norm1(f_v), return_attn=True)
			
			if self.is_bg_cls:
				vis_attn = attn[:, :, 0, 1:-(1+self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)
			else:
				vis_attn = attn[:, :, 0, 1:-(self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)

			f_v = f_v + blk.drop_path1(blk.ls1(x))
			
			x, attn = blk.attn(blk.norm1(f_a), return_attn=True)
			
			if self.is_bg_cls:
				aud_attn = attn[:, :, 0, 1:-(1+self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)
			else:
				aud_attn = attn[:, :, 0, 1:-(self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)

			f_a = f_a + blk.drop_path1(blk.ls1(x))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)
			
				f_a_res = self.audio_adapter_blocks_p2[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
	
			f_v = f_v + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_v))))
			f_a = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_a))))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)

			if self.is_bg_cls:
				vis_patch =  f_v[:, 1:-(1+self.n_vis_tokens+self.n_shared_tokens), :]
				aud_patch = f_a[:, 1:-(1+self.n_audio_tokens+self.n_shared_tokens), :]
			else:
				vis_patch =  f_v[:, 1:-(self.n_vis_tokens+self.n_shared_tokens), :]
				aud_patch = f_a[:, 1:-(self.n_audio_tokens+self.n_shared_tokens), :]

			if self.contrastive == 'blockwise_sep':
				audio_feat = F.normalize(self.audio_proj[layer_count](aud_attn @ aud_patch).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj[layer_count](vis_attn @ vis_patch).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'blockwise_cmn':
				audio_feat = F.normalize(self.audio_proj(aud_attn @ aud_patch).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj(vis_attn @ vis_patch).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'final' and i == (self.total_layers-1):
				audio_feat = F.normalize(self.audio_proj(aud_attn @ aud_patch).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj(vis_attn @ vis_patch).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			if self.layerwise_token is not None and i != (self.total_layers-1):
				f_a = f_a[:, :-(self.n_vis_tokens+self.n_shared_tokens) , :]
				f_v = f_v[:, :-(self.n_audio_tokens+self.n_shared_tokens), :]

			layer_count += 1

		if self.contrastive in ["blockwise_sep", "blockwise_cmn"]:
			cnt_loss = cnt_loss/self.total_layers

		f_v = self.ViT.v.norm(f_v)
		f_a = self.ViT.v.norm(f_a)

		v_cls = f_v[:, 0:1].squeeze()
		a_cls = f_a[:, 0:1].squeeze()

		fg_feat = torch.cat((v_cls, a_cls), dim=-1)
		p_fg = self.fg_class(fg_feat)

		if self.is_bg_cls:
			v_bg_cls = f_v[:, -(1+self.n_vis_tokens+self.n_shared_tokens):-(self.n_vis_tokens+self.n_shared_tokens)].squeeze()
			a_bg_cls = f_a[:, -(1+self.n_audio_tokens+self.n_shared_tokens):-(self.n_audio_tokens+self.n_shared_tokens)].squeeze()
			bg_feat = torch.cat((v_bg_cls, a_bg_cls), dim=-1)
			p_bg = self.bg_class(bg_feat).squeeze()
			
			bg_loss = self.calc_bg_loss(p_bg, bg_label)

		fg_loss = self.calc_fg_loss(p_fg, label, bg_label)

		loss = fg_loss + bg_loss + cnt_loss

		if self.grad_mod:
			coeff_v, coeff_a = self.calc_grad_coeff(v_cls, a_cls, label, bg_label)

		return {
			'loss': loss,
			'fg_loss': fg_loss,
			'bg_loss': bg_loss if self.is_bg_cls else None,
			'cnt_loss': cnt_loss,
			'p_fg': p_fg,
			'p_bg': p_bg if self.is_bg_cls else None,
			'vis_attn': vis_attn,
			'coeff_a': coeff_a,
			'coeff_v': coeff_v
		}

	def forward(self, audio, vis, return_attn=False):
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, return_attn)


class CLS_Token_Proj_AVIT(nn.Module):
	def __init__(self, opt):
		super(CLS_Token_Proj_AVIT, self).__init__()
		
		assert opt.vis_encoder_type == 'vit', "only vit is supported for now"

		self.opt = opt
		self.ngpu = opt.ngpu
		self.rank = opt.rank

		if opt.vis_encoder_type == 'vit':
			if opt.vit_type == "tiny":
				self.ViT = my_vit('vit_tiny_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "small":
				self.ViT = my_vit(name='vit_small_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "base":
				self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "large":
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)
			else:
				print("Nothing found. TinyViT is loading")
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)

			self.dim = self.ViT.v.patch_embed.proj.out_channels

			self.fg_class = nn.Linear(self.dim * 2, opt.num_class)

		hidden_list = []
		self.total_layers = len(self.ViT.v.blocks)

		self.contrastive = opt.contrastive
		self.LSA = opt.LSA

		if opt.contrastive in ['final' or 'blockwise_cmn']:
			self.audio_proj = nn.Linear(self.dim, 512)
			self.vis_proj = nn.Linear(self.dim, 512)
			self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

		elif opt.contrastive == 'blockwise_sep':
			self.audio_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 

			self.vis_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 
			
			self.logit_scale = nn.ParameterList([
				nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
			for i in range(self.total_layers)])

		if opt.vis_encoder_type == 'vit':
			for idx_layer, my_blk in enumerate(self.ViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)

				my_blk.attn.forward = types.MethodType(attn_new_forward, my_blk.attn)

		self.lavish_adapter = opt.lavish_adapter

		if opt.lavish_adapter:
			self.audio_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

			self.audio_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

		self.unimodal_token = opt.unimodal_token
		self.multimodal_token = opt.multimodal_token
		self.layerwise_token = opt.layerwise_token # "None" or "common" or "unique"
		self.layerwise_attn = opt.layerwise_attn # "None" or "common" or "unique"

		if opt.unimodal_token:
			self.audio_tokens = nn.Parameter(torch.randn(opt.n_audio_tokens, hidden_d_size))
			self.visual_tokens = nn.Parameter(torch.randn(opt.n_vis_tokens, hidden_d_size))
		if opt.multimodal_token:
			self.shared_tokens = nn.Parameter(torch.randn(opt.n_shared_tokens, hidden_d_size))

		self.n_audio_tokens = opt.n_audio_tokens
		self.n_vis_tokens = opt.n_vis_tokens
		self.n_shared_tokens = opt.n_shared_tokens

		self.audio_attn = nn.Identity()
		self.visual_attn = nn.Identity()
		self.shared_attn = nn.Identity()

		if opt.unimodal_token and opt.LSA:
			self.audio_attn = SelfAttention(hidden_d_size)
			self.visual_attn = SelfAttention(hidden_d_size)
		if opt.multimodal_token and opt.LSA:
			self.shared_attn = SelfAttention(hidden_d_size)

		self.is_bg_cls = opt.bg_cls
		if opt.bg_cls:
			self.bg_cls_token = nn.Parameter(torch.randn(1, hidden_d_size))
			self.bg_class = nn.Linear(self.dim * 2, 1)
			self.bg_label = opt.bg_label

		self.grad_mod = opt.grad_mod

	def calc_cnt_loss(self, image_feat, audio_feat, bg_label, layer_count):
		gather_image_feat = [torch.zeros_like(image_feat) for _ in range(self.ngpu)]
		gather_audio_feat = [torch.zeros_like(audio_feat) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			gather_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(gather_image_feat, image_feat)
		dist.all_gather(gather_audio_feat, audio_feat)
		if self.is_bg_cls:
			dist.all_gather(gather_bg_label, bg_label)
		
		gather_image_feat[self.rank] = image_feat
		gather_audio_feat[self.rank] = audio_feat

		all_image_feat = torch.cat(gather_image_feat, dim=0)
		all_audio_feat = torch.cat(gather_audio_feat, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(gather_bg_label, dim=0)

		if self.is_bg_cls:
			mask = torch.where(all_bg_label == 0)[0]
			all_image_feat = all_image_feat[mask]
			all_audio_feat = all_audio_feat[mask]
		
		label = torch.arange(len(all_image_feat)).to(image_feat.device)
		
		if self.contrastive in ['blockwise_sep', 'blockwise_cmn']:
			pred_img = (1/0.07) * (all_image_feat @ all_audio_feat.T)	
			pred_aud = pred_img.T
		else:
			pred_img = (1/0.07)* (all_image_feat @ all_audio_feat.T)	
			pred_aud = pred_img.T

		loss = (F.cross_entropy(pred_img, label) + F.cross_entropy(pred_aud, label))/2

		return loss

	def calc_grad_coeff(self, v_cls, a_cls, label, bg_label):
		'''
		return coeff_v, coeff_a
		'''
		weight_size = self.fg_class.weight.size(1)
		out_v = (torch.mm(v_cls, torch.transpose(self.fg_class.weight[:, weight_size // 2:], 0, 1))
					+ self.fg_class.bias / 2)

		out_a = (torch.mm(a_cls, torch.transpose(self.fg_class.weight[:, :weight_size // 2], 0, 1))
					+ self.fg_class.bias / 2)

		# Modulation starts here !
		all_out_v = [torch.zeros_like(out_v) for _ in range(self.ngpu)]
		all_out_a = [torch.zeros_like(out_a) for _ in range(self.ngpu)]
		all_label = [torch.zeros_like(label) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			all_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(all_out_v, out_v)
		dist.all_gather(all_out_a, out_a)
		dist.all_gather(all_label, label)
		if self.is_bg_cls:
			dist.all_gather(all_bg_label, bg_label)

		all_out_v = torch.cat(all_out_v, dim=0)
		all_out_a = torch.cat(all_out_a, dim=0)
		all_label = torch.cat(all_label, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(all_bg_label, dim=0)

			mask = torch.where(all_bg_label == 0)[0]
			all_out_v = all_out_v[mask]
			all_out_a = all_out_a[mask]
			all_label = all_label[mask]

		score_v = sum([F.softmax(all_out_v)[i][all_label[i]] for i in range(all_out_v.size(0))])
		score_a = sum([F.softmax(all_out_a)[i][all_label[i]] for i in range(all_out_a.size(0))])

		ratio_v = score_v / score_a
		ratio_a = 1 / ratio_v

		if ratio_v > 1:
			coeff_v = 1 - F.tanh(0.1 * F.relu(ratio_v))
			coeff_a = 1
		else:
			coeff_a = 1 - F.tanh(0.1 * F.relu(ratio_a))
			coeff_v = 1

		return coeff_v, coeff_a


	def calc_fg_loss(self, p_fg, label, bg_label):
		if self.is_bg_cls:
			mask = torch.where(bg_label == 0)[0]

			label = label[mask]
			p_fg = p_fg[mask]

		loss = F.cross_entropy(p_fg, label)

		return loss

	def calc_bg_loss(self, p_bg, bg_label):
		loss = F.binary_cross_entropy_with_logits(p_bg, bg_label.float())
		return loss 

	def forward_vit(self, audio, vis, label, return_attn=True):
		b, t, c, w, h = vis.shape

		label = label.view(-1)

		if self.is_bg_cls:
			bg_label = (label == self.bg_label).long()
		else:
			bg_label = None

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)

		if self.is_bg_cls:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token)
		else:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True)

		bs, dim, wa, ha = patch_info_audio
		bs, dim, wv, wv = patch_info_vis

		layer_count = 0
		cnt_loss = 0
		fg_loss = 0
		bg_loss = 0

		for i, blk in enumerate(self.ViT.v.blocks):
			if i == 0 or self.layerwise_token in ["unique", "common"]:
				if self.unimodal_token:
					audio_tokens = repeat(self.audio_tokens, 'len dim -> b len dim', b=f_a.shape[0])
					visual_tokens = repeat(self.visual_tokens, 'len dim -> b len dim', b=f_v.shape[0])

					audio_tokens = self.audio_attn(audio_tokens)
					visual_tokens = self.visual_attn(visual_tokens)

					f_a = torch.cat([
						f_a,
						audio_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						visual_tokens
					], dim=1)

				if self.multimodal_token:
					shared_tokens = repeat(self.shared_tokens, 'len dim -> b len dim', b=f_v.shape[0])
					shared_tokens = self.shared_attn(shared_tokens)
			
					f_a = torch.cat([
						f_a,
						shared_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						shared_tokens
					], dim=1)

			if self.lavish_adapter:
				f_a_res = self.audio_adapter_blocks_p1[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))

			x, attn = blk.attn(blk.norm1(f_v), return_attn=True)
			
			if self.is_bg_cls:
				vis_attn = attn[:, :, 0, 1:-(1+self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)
			else:
				vis_attn = attn[:, :, 0, 1:-(self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)

			f_v = f_v + blk.drop_path1(blk.ls1(x))
		
			x, attn = blk.attn(blk.norm1(f_a), return_attn=True)
			
			if self.is_bg_cls:
				aud_attn = attn[:, :, 0, 1:-(1+self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)
			else:
				aud_attn = attn[:, :, 0, 1:-(self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)

			f_a = f_a + blk.drop_path1(blk.ls1(x))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)
			
				f_a_res = self.audio_adapter_blocks_p2[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
	
			f_v = f_v + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_v))))
			f_a = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_a))))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)

			if self.is_bg_cls:
				vis_patch =  f_v[:, 1:-(1+self.n_vis_tokens+self.n_shared_tokens), :]
				aud_patch = f_a[:, 1:-(1+self.n_audio_tokens+self.n_shared_tokens), :]
			else:
				vis_patch =  f_v[:, 1:-(self.n_vis_tokens+self.n_shared_tokens), :]
				aud_patch = f_a[:, 1:-(self.n_audio_tokens+self.n_shared_tokens), :]

			audio_token = f_a[:, 0, :]
			image_token = f_v[:, 0, :]

			if self.contrastive == 'blockwise_sep':
				audio_feat = F.normalize(self.audio_proj[layer_count](audio_token).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj[layer_count](image_token).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'blockwise_cmn':
				audio_feat = F.normalize(self.audio_proj(audio_token).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj(image_token).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'final' and i == (self.total_layers-1):
				audio_feat = F.normalize(self.audio_proj(audio_token).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj(image_token).squeeze(), dim=1).float()

				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			if self.layerwise_token is not None and i != (self.total_layers-1):
				f_a = f_a[:, :-(self.n_vis_tokens+self.n_shared_tokens) , :]
				f_v = f_v[:, :-(self.n_audio_tokens+self.n_shared_tokens), :]

			layer_count += 1

		if self.contrastive in ["blockwise_sep", "blockwise_cmn"]:
			cnt_loss = cnt_loss/self.total_layers

		f_v = self.ViT.v.norm(f_v)
		f_a = self.ViT.v.norm(f_a)

		v_cls = f_v[:, 0:1].squeeze()
		a_cls = f_a[:, 0:1].squeeze()

		fg_feat = torch.cat((v_cls, a_cls), dim=-1)
		p_fg = self.fg_class(fg_feat)

		if self.is_bg_cls:
			v_bg_cls = f_v[:, -(1+self.n_vis_tokens+self.n_shared_tokens):-(self.n_vis_tokens+self.n_shared_tokens)].squeeze()
			a_bg_cls = f_a[:, -(1+self.n_audio_tokens+self.n_shared_tokens):-(self.n_audio_tokens+self.n_shared_tokens)].squeeze()
			bg_feat = torch.cat((v_bg_cls, a_bg_cls), dim=-1)
			p_bg = self.bg_class(bg_feat).squeeze()
			
			bg_loss = self.calc_bg_loss(p_bg, bg_label)

		fg_loss = self.calc_fg_loss(p_fg, label, bg_label)

		loss = fg_loss + bg_loss + cnt_loss

		if self.grad_mod:
			coeff_v, coeff_a = self.calc_grad_coeff(v_cls, a_cls, label, bg_label)
		else:
			coeff_v, coeff_a = None, None

		return {
			'loss': loss,
			'fg_loss': fg_loss,
			'bg_loss': bg_loss if self.is_bg_cls else None,
			'cnt_loss': cnt_loss,
			'p_fg': p_fg,
			'p_bg': p_bg if self.is_bg_cls else None,
			'vis_attn': vis_attn,
			'coeff_a': coeff_a,
			'coeff_v': coeff_v
		}

	def forward(self, audio, vis, return_attn=False):
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, return_attn)



class CNT_Token_Proj_AVIT(nn.Module):
	def __init__(self, opt):
		super(CNT_Token_Proj_AVIT, self).__init__()
		
		assert opt.vis_encoder_type == 'vit', "only vit is supported for now"

		self.opt = opt
		self.ngpu = opt.ngpu
		self.rank = opt.rank

		if opt.vis_encoder_type == 'vit':
			if opt.vit_type == "tiny":
				self.ViT = my_vit('vit_tiny_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "small":
				self.ViT = my_vit(name='vit_small_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "base":
				self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "large":
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)
			else:
				print("Nothing found. TinyViT is loading")
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)

			self.dim = self.ViT.v.patch_embed.proj.out_channels

			self.fg_class = nn.Linear(self.dim * 2, opt.num_class)

		hidden_list = []
		self.total_layers = len(self.ViT.v.blocks)

		self.contrastive = opt.contrastive
		self.LSA = opt.LSA

		if opt.contrastive in ['final', 'blockwise_cmn']:
			self.audio_proj = nn.Linear(self.dim, 512)
			self.vis_proj = nn.Linear(self.dim, 512)
			self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

		elif opt.contrastive == 'blockwise_sep':
			self.audio_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 

			self.vis_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 
			
			self.logit_scale = nn.ParameterList([
				nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
			for i in range(self.total_layers)])

		if opt.vis_encoder_type == 'vit':
			for idx_layer, my_blk in enumerate(self.ViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)

				my_blk.attn.forward = types.MethodType(attn_new_forward, my_blk.attn)

		self.lavish_adapter = opt.lavish_adapter

		if opt.lavish_adapter:
			self.audio_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

			self.audio_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

		self.unimodal_token = opt.unimodal_token
		self.multimodal_token = opt.multimodal_token
		self.layerwise_token = opt.layerwise_token # "None" or "common" or "unique"
		self.layerwise_attn = opt.layerwise_attn # "None" or "common" or "unique"

		self.cnt_audio_token = nn.Parameter(torch.randn(1, hidden_d_size))
		self.cnt_visual_token = nn.Parameter(torch.randn(1, hidden_d_size))

		if opt.unimodal_token:
			self.audio_tokens = nn.Parameter(torch.randn(opt.n_audio_tokens, hidden_d_size))
			self.visual_tokens = nn.Parameter(torch.randn(opt.n_vis_tokens, hidden_d_size))
		if opt.multimodal_token:
			self.shared_tokens = nn.Parameter(torch.randn(opt.n_shared_tokens, hidden_d_size))

		self.n_audio_tokens = opt.n_audio_tokens
		self.n_vis_tokens = opt.n_vis_tokens
		self.n_shared_tokens = opt.n_shared_tokens

		self.audio_attn = nn.Identity()
		self.visual_attn = nn.Identity()
		self.shared_attn = nn.Identity()

		if opt.unimodal_token and opt.LSA:
			self.audio_attn = SelfAttention(hidden_d_size)
			self.visual_attn = SelfAttention(hidden_d_size)
		if opt.multimodal_token and opt.LSA:
			self.shared_attn = SelfAttention(hidden_d_size)
		
		self.is_bg_cls = opt.bg_cls
		if opt.bg_cls:
			self.bg_cls_token = nn.Parameter(torch.randn(1, hidden_d_size))
			self.bg_class = nn.Linear(self.dim * 2, 1)
			self.bg_label = opt.bg_label

		self.grad_mod = opt.grad_mod

	def calc_cnt_loss(self, image_feat, audio_feat, bg_label, layer_count):
		gather_image_feat = [torch.zeros_like(image_feat) for _ in range(self.ngpu)]
		gather_audio_feat = [torch.zeros_like(audio_feat) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			gather_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(gather_image_feat, image_feat)
		dist.all_gather(gather_audio_feat, audio_feat)
		if self.is_bg_cls:
			dist.all_gather(gather_bg_label, bg_label)
		
		gather_image_feat[self.rank] = image_feat
		gather_audio_feat[self.rank] = audio_feat

		all_image_feat = torch.cat(gather_image_feat, dim=0)
		all_audio_feat = torch.cat(gather_audio_feat, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(gather_bg_label, dim=0)

		if self.is_bg_cls:
			mask = torch.where(all_bg_label == 0)[0]
			all_image_feat = all_image_feat[mask]
			all_audio_feat = all_audio_feat[mask]
		
		label = torch.arange(len(all_image_feat)).to(image_feat.device)
		
		if self.contrastive in ['blockwise_sep', 'blockwise_cmn']:
			pred_img = (1/0.07) * (all_image_feat @ all_audio_feat.T)	
			pred_aud = pred_img.T
		else:
			pred_img = (1/0.07)* (all_image_feat @ all_audio_feat.T)	
			pred_aud = pred_img.T

		loss = (F.cross_entropy(pred_img, label) + F.cross_entropy(pred_aud, label))/2

		return loss

	def calc_grad_coeff(self, v_cls, a_cls, label, bg_label):
		'''
		return coeff_v, coeff_a
		'''
		weight_size = self.fg_class.weight.size(1)
		out_v = (torch.mm(v_cls, torch.transpose(self.fg_class.weight[:, weight_size // 2:], 0, 1))
					+ self.fg_class.bias / 2)

		out_a = (torch.mm(a_cls, torch.transpose(self.fg_class.weight[:, :weight_size // 2], 0, 1))
					+ self.fg_class.bias / 2)

		# Modulation starts here !
		all_out_v = [torch.zeros_like(out_v) for _ in range(self.ngpu)]
		all_out_a = [torch.zeros_like(out_a) for _ in range(self.ngpu)]
		all_label = [torch.zeros_like(label) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			all_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(all_out_v, out_v)
		dist.all_gather(all_out_a, out_a)
		dist.all_gather(all_label, label)
		if self.is_bg_cls:
			dist.all_gather(all_bg_label, bg_label)

		all_out_v = torch.cat(all_out_v, dim=0)
		all_out_a = torch.cat(all_out_a, dim=0)
		all_label = torch.cat(all_label, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(all_bg_label, dim=0)

			mask = torch.where(all_bg_label == 0)[0]
			all_out_v = all_out_v[mask]
			all_out_a = all_out_a[mask]
			all_label = all_label[mask]

		score_v = sum([F.softmax(all_out_v)[i][all_label[i]] for i in range(all_out_v.size(0))])
		score_a = sum([F.softmax(all_out_a)[i][all_label[i]] for i in range(all_out_a.size(0))])

		ratio_v = score_v / score_a
		ratio_a = 1 / ratio_v

		if ratio_v > 1:
			coeff_v = 1 - F.tanh(0.1 * F.relu(ratio_v))
			coeff_a = 1
		else:
			coeff_a = 1 - F.tanh(0.1 * F.relu(ratio_a))
			coeff_v = 1

		return coeff_v, coeff_a


	def calc_fg_loss(self, p_fg, label, bg_label):
		if self.is_bg_cls:
			mask = torch.where(bg_label == 0)[0]

			label = label[mask]
			p_fg = p_fg[mask]

		loss = F.cross_entropy(p_fg, label)

		return loss

	def calc_bg_loss(self, p_bg, bg_label):
		loss = F.binary_cross_entropy_with_logits(p_bg, bg_label.float())
		return loss 

	def forward_vit(self, audio, vis, label, return_attn=True):
		b, t, c, w, h = vis.shape

		label = label.view(-1)

		if self.is_bg_cls:
			bg_label = (label == self.bg_label).long()
		else:
			bg_label = None

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)

		if self.is_bg_cls:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token, cnt_token=self.cnt_audio_token)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token, cnt_token=self.cnt_visual_token)
		else:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True, cnt_token=self.cnt_audio_token)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True, cnt_token=self.cnt_visual_token)

		bs, dim, wa, ha = patch_info_audio
		bs, dim, wv, wv = patch_info_vis

		layer_count = 0
		cnt_loss = 0
		fg_loss = 0
		bg_loss = 0

		for i, blk in enumerate(self.ViT.v.blocks):
			if i == 0 or self.layerwise_token in ["unique", "common"]:
				if self.unimodal_token:
					audio_tokens = repeat(self.audio_tokens, 'len dim -> b len dim', b=f_a.shape[0])
					visual_tokens = repeat(self.visual_tokens, 'len dim -> b len dim', b=f_v.shape[0])

					audio_tokens = self.audio_attn(audio_tokens)
					visual_tokens = self.visual_attn(visual_tokens)

					f_a = torch.cat([
						f_a,
						audio_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						visual_tokens
					], dim=1)

				if self.multimodal_token:
					shared_tokens = repeat(self.shared_tokens, 'len dim -> b len dim', b=f_v.shape[0])
					shared_tokens = self.shared_attn(shared_tokens)
			
					f_a = torch.cat([
						f_a,
						shared_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						shared_tokens
					], dim=1)

			if self.lavish_adapter:
				f_a_res = self.audio_adapter_blocks_p1[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))

			x, attn = blk.attn(blk.norm1(f_v), return_attn=True)
			
			if self.is_bg_cls:
				vis_attn = attn[:, :, 0, 1:-(1+self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)
			else:
				vis_attn = attn[:, :, 0, 1:-(self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)

			f_v = f_v + blk.drop_path1(blk.ls1(x))
		
			x, attn = blk.attn(blk.norm1(f_a), return_attn=True)
			
			if self.is_bg_cls:
				aud_attn = attn[:, :, 0, 1:-(1+self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)
			else:
				aud_attn = attn[:, :, 0, 1:-(self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)

			f_a = f_a + blk.drop_path1(blk.ls1(x))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)
			
				f_a_res = self.audio_adapter_blocks_p2[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
	
			f_v = f_v + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_v))))
			f_a = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_a))))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)

			if self.is_bg_cls:
				vis_patch =  f_v[:, 1:-(1+self.n_vis_tokens+self.n_shared_tokens), :]
				aud_patch = f_a[:, 1:-(1+self.n_audio_tokens+self.n_shared_tokens), :]
			else:
				vis_patch =  f_v[:, 1:-(self.n_vis_tokens+self.n_shared_tokens), :]
				aud_patch = f_a[:, 1:-(self.n_audio_tokens+self.n_shared_tokens), :]

			v_cnt_feat = f_v[:, -(1+int(self.is_bg_cls)+self.n_vis_tokens+self.n_shared_tokens):-(int(self.is_bg_cls)+self.n_vis_tokens+self.n_shared_tokens)].squeeze()
			a_cnt_feat = f_a[:, -(1+int(self.is_bg_cls)+self.n_audio_tokens+self.n_shared_tokens):-(+int(self.is_bg_cls)+self.n_audio_tokens+self.n_shared_tokens)].squeeze()

			if self.contrastive == 'blockwise_sep':
				audio_feat = F.normalize(self.audio_proj[layer_count](a_cnt_feat).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj[layer_count](v_cnt_feat).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'blockwise_cmn':
				audio_feat = F.normalize(self.audio_proj(a_cnt_feat), dim=1).squeeze().float()
				image_feat = F.normalize(self.vis_proj(v_cnt_feat), dim=1).squeeze().float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'final' and i == (self.total_layers-1):
				audio_feat = F.normalize(self.audio_proj(a_cnt_feat), dim=1).squeeze().float()
				image_feat = F.normalize(self.vis_proj(v_cnt_feat), dim=1).squeeze().float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			if self.layerwise_token is not None and i != (self.total_layers-1):
				f_a = f_a[:, :-(self.n_vis_tokens+self.n_shared_tokens) , :]
				f_v = f_v[:, :-(self.n_audio_tokens+self.n_shared_tokens), :]

			layer_count += 1

		if self.contrastive in ["blockwise_sep", "blockwise_cmn"]:
			cnt_loss = cnt_loss/self.total_layers

		f_v = self.ViT.v.norm(f_v)
		f_a = self.ViT.v.norm(f_a)

		v_cls = f_v[:, 0:1].squeeze()
		a_cls = f_a[:, 0:1].squeeze()

		fg_feat = torch.cat((v_cls, a_cls), dim=-1)
		p_fg = self.fg_class(fg_feat)

		if self.is_bg_cls:
			v_bg_cls = f_v[:, -(1+self.n_vis_tokens+self.n_shared_tokens):-(self.n_vis_tokens+self.n_shared_tokens)].squeeze()
			a_bg_cls = f_a[:, -(1+self.n_audio_tokens+self.n_shared_tokens):-(self.n_audio_tokens+self.n_shared_tokens)].squeeze()
			bg_feat = torch.cat((v_bg_cls, a_bg_cls), dim=-1)
			p_bg = self.bg_class(bg_feat).squeeze()
			
			bg_loss = self.calc_bg_loss(p_bg, bg_label)

		fg_loss = self.calc_fg_loss(p_fg, label, bg_label)

		loss = fg_loss + bg_loss + cnt_loss

		if self.grad_mod:
			coeff_v, coeff_a = self.calc_grad_coeff(v_cls, a_cls, label, bg_label)
		else:
			coeff_v, coeff_a = None, None

		return {
			'loss': loss,
			'fg_loss': fg_loss,
			'bg_loss': bg_loss if self.is_bg_cls else None,
			'cnt_loss': cnt_loss,
			'p_fg': p_fg,
			'p_bg': p_bg if self.is_bg_cls else None,
			'vis_attn': vis_attn,
			'coeff_a': coeff_a,
			'coeff_v': coeff_v
		}

	def forward(self, audio, vis, return_attn=False):
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, return_attn)


class Mean_Pooled_Proj_AVIT(nn.Module):
	def __init__(self, opt):
		super(Mean_Pooled_Proj_AVIT, self).__init__()
		
		assert opt.vis_encoder_type == 'vit', "only vit is supported for now"

		self.opt = opt
		self.ngpu = opt.ngpu
		self.rank = opt.rank

		if opt.vis_encoder_type == 'vit':
			if opt.vit_type == "tiny":
				self.ViT = my_vit('vit_tiny_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "small":
				self.ViT = my_vit(name='vit_small_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "base":
				self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "large":
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)
			else:
				print("Nothing found. TinyViT is loading")
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)

			self.dim = self.ViT.v.patch_embed.proj.out_channels

			self.fg_class = nn.Linear(self.dim * 2, opt.num_class)

		hidden_list = []
		self.total_layers = len(self.ViT.v.blocks)

		self.contrastive = opt.contrastive
		self.LSA = opt.LSA

		if opt.contrastive in ['final' or 'blockwise_cmn']:
			self.audio_proj = nn.Linear(self.dim, 512)
			self.vis_proj = nn.Linear(self.dim, 512)
			self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

		elif opt.contrastive == 'blockwise_sep':
			self.audio_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 

			self.vis_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 
			
			self.logit_scale = nn.ParameterList([
				nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
			for i in range(self.total_layers)])

		if opt.vis_encoder_type == 'vit':
			for idx_layer, my_blk in enumerate(self.ViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)

				my_blk.attn.forward = types.MethodType(attn_new_forward, my_blk.attn)

		self.lavish_adapter = opt.lavish_adapter

		if opt.lavish_adapter:
			self.audio_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

			self.audio_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

		self.unimodal_token = opt.unimodal_token
		self.multimodal_token = opt.multimodal_token
		self.layerwise_token = opt.layerwise_token # "None" or "common" or "unique"
		self.layerwise_attn = opt.layerwise_attn # "None" or "common" or "unique"

		if opt.unimodal_token:
			self.audio_tokens = nn.Parameter(torch.randn(opt.n_audio_tokens, hidden_d_size))
			self.visual_tokens = nn.Parameter(torch.randn(opt.n_vis_tokens, hidden_d_size))
		if opt.multimodal_token:
			self.shared_tokens = nn.Parameter(torch.randn(opt.n_shared_tokens, hidden_d_size))

		self.n_audio_tokens = opt.n_audio_tokens
		self.n_vis_tokens = opt.n_vis_tokens
		self.n_shared_tokens = opt.n_shared_tokens

		self.audio_attn = nn.Identity()
		self.visual_attn = nn.Identity()
		self.shared_attn = nn.Identity()

		if opt.unimodal_token and opt.LSA:
			self.audio_attn = SelfAttention(hidden_d_size)
			self.visual_attn = SelfAttention(hidden_d_size)
		if opt.multimodal_token and opt.LSA:
			self.shared_attn = SelfAttention(hidden_d_size)
		
		self.is_bg_cls = opt.bg_cls
		if opt.bg_cls:
			self.bg_cls_token = nn.Parameter(torch.randn(1, hidden_d_size))
			self.bg_class = nn.Linear(self.dim * 2, 1)
			self.bg_label = opt.bg_label

		self.grad_mod = opt.grad_mod

	def calc_cnt_loss(self, image_feat, audio_feat, bg_label, layer_count):
		gather_image_feat = [torch.zeros_like(image_feat) for _ in range(self.ngpu)]
		gather_audio_feat = [torch.zeros_like(audio_feat) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			gather_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(gather_image_feat, image_feat)
		dist.all_gather(gather_audio_feat, audio_feat)
		if self.is_bg_cls:
			dist.all_gather(gather_bg_label, bg_label)
		
		gather_image_feat[self.rank] = image_feat
		gather_audio_feat[self.rank] = audio_feat

		all_image_feat = torch.cat(gather_image_feat, dim=0)
		all_audio_feat = torch.cat(gather_audio_feat, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(gather_bg_label, dim=0)

		if self.is_bg_cls:
			mask = torch.where(all_bg_label == 0)[0]
			all_image_feat = all_image_feat[mask]
			all_audio_feat = all_audio_feat[mask]
		
		label = torch.arange(len(all_image_feat)).to(image_feat.device)
		
		if self.contrastive in ['blockwise_sep', 'blockwise_cmn']:
			pred_img = (1/0.07) * (all_image_feat @ all_audio_feat.T)	
			pred_aud = pred_img.T
		else:
			pred_img = (1/0.07)* (all_image_feat @ all_audio_feat.T)	
			pred_aud = pred_img.T

		loss = (F.cross_entropy(pred_img, label) + F.cross_entropy(pred_aud, label))/2

		return loss

	def calc_grad_coeff(self, v_cls, a_cls, label, bg_label):
		'''
		return coeff_v, coeff_a
		'''
		weight_size = self.fg_class.weight.size(1)
		out_v = (torch.mm(v_cls, torch.transpose(self.fg_class.weight[:, weight_size // 2:], 0, 1))
					+ self.fg_class.bias / 2)

		out_a = (torch.mm(a_cls, torch.transpose(self.fg_class.weight[:, :weight_size // 2], 0, 1))
					+ self.fg_class.bias / 2)

		# Modulation starts here !
		all_out_v = [torch.zeros_like(out_v) for _ in range(self.ngpu)]
		all_out_a = [torch.zeros_like(out_a) for _ in range(self.ngpu)]
		all_label = [torch.zeros_like(label) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			all_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(all_out_v, out_v)
		dist.all_gather(all_out_a, out_a)
		dist.all_gather(all_label, label)
		if self.is_bg_cls:
			dist.all_gather(all_bg_label, bg_label)

		all_out_v = torch.cat(all_out_v, dim=0)
		all_out_a = torch.cat(all_out_a, dim=0)
		all_label = torch.cat(all_label, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(all_bg_label, dim=0)

			mask = torch.where(all_bg_label == 0)[0]
			all_out_v = all_out_v[mask]
			all_out_a = all_out_a[mask]
			all_label = all_label[mask]

		score_v = sum([F.softmax(all_out_v)[i][all_label[i]] for i in range(all_out_v.size(0))])
		score_a = sum([F.softmax(all_out_a)[i][all_label[i]] for i in range(all_out_a.size(0))])

		ratio_v = score_v / score_a
		ratio_a = 1 / ratio_v

		if ratio_v > 1:
			coeff_v = 1 - F.tanh(0.1 * F.relu(ratio_v))
			coeff_a = 1
		else:
			coeff_a = 1 - F.tanh(0.1 * F.relu(ratio_a))
			coeff_v = 1

		return coeff_v, coeff_a


	def calc_fg_loss(self, p_fg, label, bg_label):
		if self.is_bg_cls:
			mask = torch.where(bg_label == 0)[0]

			label = label[mask]
			p_fg = p_fg[mask]

		loss = F.cross_entropy(p_fg, label)

		return loss

	def calc_bg_loss(self, p_bg, bg_label):
		loss = F.binary_cross_entropy_with_logits(p_bg, bg_label.float())
		return loss 

	def forward_vit(self, audio, vis, label, return_attn=True):
		b, t, c, w, h = vis.shape

		label = label.view(-1)

		if self.is_bg_cls:
			bg_label = (label == self.bg_label).long()
		else:
			bg_label = None

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)

		if self.is_bg_cls:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token)
		else:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True)

		bs, dim, wa, ha = patch_info_audio
		bs, dim, wv, wv = patch_info_vis

		layer_count = 0
		cnt_loss = 0
		fg_loss = 0
		bg_loss = 0

		for i, blk in enumerate(self.ViT.v.blocks):
			if i == 0 or self.layerwise_token in ["unique", "common"]:
				if self.unimodal_token:
					audio_tokens = repeat(self.audio_tokens, 'len dim -> b len dim', b=f_a.shape[0])
					visual_tokens = repeat(self.visual_tokens, 'len dim -> b len dim', b=f_v.shape[0])

					audio_tokens = self.audio_attn(audio_tokens)
					visual_tokens = self.visual_attn(visual_tokens)

					f_a = torch.cat([
						f_a,
						audio_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						visual_tokens
					], dim=1)

				if self.multimodal_token:
					shared_tokens = repeat(self.shared_tokens, 'len dim -> b len dim', b=f_v.shape[0])
					shared_tokens = self.shared_attn(shared_tokens)
			
					f_a = torch.cat([
						f_a,
						shared_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						shared_tokens
					], dim=1)

			if self.lavish_adapter:
				f_a_res = self.audio_adapter_blocks_p1[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))

			x, attn = blk.attn(blk.norm1(f_v), return_attn=True)
			
			if self.is_bg_cls:
				vis_attn = attn[:, :, 0, 1:-(1+self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)
			else:
				vis_attn = attn[:, :, 0, 1:-(self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)

			f_v = f_v + blk.drop_path1(blk.ls1(x))
		
			x, attn = blk.attn(blk.norm1(f_a), return_attn=True)
			
			if self.is_bg_cls:
				aud_attn = attn[:, :, 0, 1:-(1+self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)
			else:
				aud_attn = attn[:, :, 0, 1:-(self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)

			f_a = f_a + blk.drop_path1(blk.ls1(x))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)
			
				f_a_res = self.audio_adapter_blocks_p2[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
	
			f_v = f_v + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_v))))
			f_a = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_a))))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)

			if self.is_bg_cls:
				vis_patch =  f_v[:, 1:-(1+self.n_vis_tokens+self.n_shared_tokens), :]
				aud_patch = f_a[:, 1:-(1+self.n_audio_tokens+self.n_shared_tokens), :]
			else:
				vis_patch =  f_v[:, 1:-(self.n_vis_tokens+self.n_shared_tokens), :]
				aud_patch = f_a[:, 1:-(self.n_audio_tokens+self.n_shared_tokens), :]

			if self.contrastive == 'blockwise_sep':
				audio_feat = F.normalize(self.audio_proj[layer_count](aud_patch.mean(dim=1)).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj[layer_count](vis_patch.mean(dim=1)).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'blockwise_cmn':
				audio_feat = F.normalize(self.audio_proj(aud_patch.mean(dim=1)).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj(vis_patch.mean(dim=1)).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'final' and i == (self.total_layers-1):
				audio_feat = F.normalize(self.audio_proj(aud_patch.mean(dim=1)).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj(vis_patch.mean(dim=1)).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			if self.layerwise_token is not None and i != (self.total_layers-1):
				f_a = f_a[:, :-(self.n_vis_tokens+self.n_shared_tokens) , :]
				f_v = f_v[:, :-(self.n_audio_tokens+self.n_shared_tokens), :]

			layer_count += 1

		if self.contrastive in ["blockwise_sep", "blockwise_cmn"]:
			cnt_loss = cnt_loss/self.total_layers

		f_v = self.ViT.v.norm(f_v)
		f_a = self.ViT.v.norm(f_a)

		v_cls = f_v[:, 0:1].squeeze()
		a_cls = f_a[:, 0:1].squeeze()

		fg_feat = torch.cat((v_cls, a_cls), dim=-1)
		p_fg = self.fg_class(fg_feat)

		if self.is_bg_cls:
			v_bg_cls = f_v[:, -(1+self.n_vis_tokens+self.n_shared_tokens):-(self.n_vis_tokens+self.n_shared_tokens)].squeeze()
			a_bg_cls = f_a[:, -(1+self.n_audio_tokens+self.n_shared_tokens):-(self.n_audio_tokens+self.n_shared_tokens)].squeeze()
			bg_feat = torch.cat((v_bg_cls, a_bg_cls), dim=-1)
			p_bg = self.bg_class(bg_feat).squeeze()
			
			bg_loss = self.calc_bg_loss(p_bg, bg_label)

		fg_loss = self.calc_fg_loss(p_fg, label, bg_label)

		loss = fg_loss + bg_loss + cnt_loss

		if self.grad_mod:
			coeff_v, coeff_a = self.calc_grad_coeff(v_cls, a_cls, label, bg_label)
		else:
			coeff_v, coeff_a = None, None

		return {
			'loss': loss,
			'fg_loss': fg_loss,
			'bg_loss': bg_loss if self.is_bg_cls else None,
			'cnt_loss': cnt_loss,
			'p_fg': p_fg,
			'p_bg': p_bg if self.is_bg_cls else None,
			'vis_attn': vis_attn,
			'coeff_a': coeff_a,
			'coeff_v': coeff_v
		}

	def forward(self, audio, vis, return_attn=False):
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, return_attn)


class Attn_Pooled_Proj_AVIT(nn.Module):
	def __init__(self, opt):
		super(Attn_Pooled_Proj_AVIT, self).__init__()
		
		assert opt.vis_encoder_type == 'vit', "only vit is supported for now"

		self.opt = opt
		self.ngpu = opt.ngpu
		self.rank = opt.rank

		if opt.vis_encoder_type == 'vit':
			if opt.vit_type == "tiny":
				self.ViT = my_vit('vit_tiny_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "small":
				self.ViT = my_vit(name='vit_small_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "base":
				self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "large":
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)
			else:
				print("Nothing found. TinyViT is loading")
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)

			self.dim = self.ViT.v.patch_embed.proj.out_channels

			self.fg_class = nn.Linear(self.dim * 2, opt.num_class)

		hidden_list = []
		self.total_layers = len(self.ViT.v.blocks)

		self.contrastive = opt.contrastive
		self.LSA = opt.LSA

		if opt.contrastive in ['final' or 'blockwise_cmn']:
			self.audio_proj = nn.Linear(self.dim, 512)
			self.vis_proj = nn.Linear(self.dim, 512)
			self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

		elif opt.contrastive == 'blockwise_sep':
			self.audio_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 

			self.vis_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 
			
			self.logit_scale = nn.ParameterList([
				nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
			for i in range(self.total_layers)])

		if opt.vis_encoder_type == 'vit':
			for idx_layer, my_blk in enumerate(self.ViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)

				my_blk.attn.forward = types.MethodType(attn_new_forward, my_blk.attn)

		self.lavish_adapter = opt.lavish_adapter

		if opt.lavish_adapter:
			self.audio_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

			self.audio_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

		self.unimodal_token = opt.unimodal_token
		self.multimodal_token = opt.multimodal_token
		self.layerwise_token = opt.layerwise_token # "None" or "common" or "unique"
		self.layerwise_attn = opt.layerwise_attn # "None" or "common" or "unique"

		if opt.unimodal_token:
			self.audio_tokens = nn.Parameter(torch.randn(opt.n_audio_tokens, hidden_d_size))
			self.visual_tokens = nn.Parameter(torch.randn(opt.n_vis_tokens, hidden_d_size))
		if opt.multimodal_token:
			self.shared_tokens = nn.Parameter(torch.randn(opt.n_shared_tokens, hidden_d_size))

		self.n_audio_tokens = opt.n_audio_tokens
		self.n_vis_tokens = opt.n_vis_tokens
		self.n_shared_tokens = opt.n_shared_tokens

		self.audio_attn = nn.Identity()
		self.visual_attn = nn.Identity()
		self.shared_attn = nn.Identity()

		if opt.unimodal_token and opt.LSA:
			self.audio_attn = SelfAttention(hidden_d_size)
			self.visual_attn = SelfAttention(hidden_d_size)
		if opt.multimodal_token and opt.LSA:
			self.shared_attn = SelfAttention(hidden_d_size)
		
		self.is_bg_cls = opt.bg_cls
		if opt.bg_cls:
			self.bg_cls_token = nn.Parameter(torch.randn(1, hidden_d_size))
			self.bg_class = nn.Linear(self.dim * 2, 1)
			self.bg_label = opt.bg_label

		self.grad_mod = opt.grad_mod

	def calc_cnt_loss(self, image_feat, audio_feat, bg_label, layer_count):
		gather_image_feat = [torch.zeros_like(image_feat) for _ in range(self.ngpu)]
		gather_audio_feat = [torch.zeros_like(audio_feat) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			gather_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(gather_image_feat, image_feat)
		dist.all_gather(gather_audio_feat, audio_feat)
		if self.is_bg_cls:
			dist.all_gather(gather_bg_label, bg_label)
		
		gather_image_feat[self.rank] = image_feat
		gather_audio_feat[self.rank] = audio_feat

		all_image_feat = torch.cat(gather_image_feat, dim=0)
		all_audio_feat = torch.cat(gather_audio_feat, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(gather_bg_label, dim=0)

		if self.is_bg_cls:
			mask = torch.where(all_bg_label == 0)[0]
			all_image_feat = all_image_feat[mask]
			all_audio_feat = all_audio_feat[mask]
		
		label = torch.arange(len(all_image_feat)).to(image_feat.device)
		
		if self.contrastive in ['blockwise_sep', 'blockwise_cmn']:
			pred_img = (1/0.07) * (all_image_feat @ all_audio_feat.T)	
			pred_aud = pred_img.T
		else:
			pred_img = (1/0.07)* (all_image_feat @ all_audio_feat.T)	
			pred_aud = pred_img.T

		loss = (F.cross_entropy(pred_img, label) + F.cross_entropy(pred_aud, label))/2

		return loss

	def calc_grad_coeff(self, v_cls, a_cls, label, bg_label):
		'''
		return coeff_v, coeff_a
		'''
		weight_size = self.fg_class.weight.size(1)
		out_v = (torch.mm(v_cls, torch.transpose(self.fg_class.weight[:, weight_size // 2:], 0, 1))
					+ self.fg_class.bias / 2)

		out_a = (torch.mm(a_cls, torch.transpose(self.fg_class.weight[:, :weight_size // 2], 0, 1))
					+ self.fg_class.bias / 2)

		# Modulation starts here !
		all_out_v = [torch.zeros_like(out_v) for _ in range(self.ngpu)]
		all_out_a = [torch.zeros_like(out_a) for _ in range(self.ngpu)]
		all_label = [torch.zeros_like(label) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			all_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(all_out_v, out_v)
		dist.all_gather(all_out_a, out_a)
		dist.all_gather(all_label, label)
		if self.is_bg_cls:
			dist.all_gather(all_bg_label, bg_label)

		all_out_v = torch.cat(all_out_v, dim=0)
		all_out_a = torch.cat(all_out_a, dim=0)
		all_label = torch.cat(all_label, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(all_bg_label, dim=0)

			mask = torch.where(all_bg_label == 0)[0]
			all_out_v = all_out_v[mask]
			all_out_a = all_out_a[mask]
			all_label = all_label[mask]

		score_v = sum([F.softmax(all_out_v)[i][all_label[i]] for i in range(all_out_v.size(0))])
		score_a = sum([F.softmax(all_out_a)[i][all_label[i]] for i in range(all_out_a.size(0))])

		ratio_v = score_v / score_a
		ratio_a = 1 / ratio_v

		if ratio_v > 1:
			coeff_v = 1 - F.tanh(0.1 * F.relu(ratio_v))
			coeff_a = 1
		else:
			coeff_a = 1 - F.tanh(0.1 * F.relu(ratio_a))
			coeff_v = 1

		return coeff_v, coeff_a


	def calc_fg_loss(self, p_fg, label, bg_label):
		if self.is_bg_cls:
			mask = torch.where(bg_label == 0)[0]

			label = label[mask]
			p_fg = p_fg[mask]

		loss = F.cross_entropy(p_fg, label)

		return loss

	def calc_bg_loss(self, p_bg, bg_label):
		loss = F.binary_cross_entropy_with_logits(p_bg, bg_label.float())
		return loss 

	def forward_vit(self, audio, vis, label, return_attn=True):
		b, t, c, w, h = vis.shape

		label = label.view(-1)

		if self.is_bg_cls:
			bg_label = (label == self.bg_label).long()
		else:
			bg_label = None

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)

		if self.is_bg_cls:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token)
		else:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True)

		bs, dim, wa, ha = patch_info_audio
		bs, dim, wv, wv = patch_info_vis

		layer_count = 0
		cnt_loss = 0
		fg_loss = 0
		bg_loss = 0

		for i, blk in enumerate(self.ViT.v.blocks):
			if i == 0 or self.layerwise_token in ["unique", "common"]:
				if self.unimodal_token:
					audio_tokens = repeat(self.audio_tokens, 'len dim -> b len dim', b=f_a.shape[0])
					visual_tokens = repeat(self.visual_tokens, 'len dim -> b len dim', b=f_v.shape[0])

					audio_tokens = self.audio_attn(audio_tokens)
					visual_tokens = self.visual_attn(visual_tokens)

					f_a = torch.cat([
						f_a,
						audio_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						visual_tokens
					], dim=1)

				if self.multimodal_token:
					shared_tokens = repeat(self.shared_tokens, 'len dim -> b len dim', b=f_v.shape[0])
					shared_tokens = self.shared_attn(shared_tokens)
			
					f_a = torch.cat([
						f_a,
						shared_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						shared_tokens
					], dim=1)

			if self.lavish_adapter:
				f_a_res = self.audio_adapter_blocks_p1[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))

			x, attn = blk.attn(blk.norm1(f_v), return_attn=True)
			
			if self.is_bg_cls:
				vis_attn = attn[:, :, 0, 1:-(1+self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)
			else:
				vis_attn = attn[:, :, 0, 1:-(self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)

			f_v = f_v + blk.drop_path1(blk.ls1(x))
		
			x, attn = blk.attn(blk.norm1(f_a), return_attn=True)
			
			if self.is_bg_cls:
				aud_attn = attn[:, :, 0, 1:-(1+self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)
			else:
				aud_attn = attn[:, :, 0, 1:-(self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)

			f_a = f_a + blk.drop_path1(blk.ls1(x))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)
			
				f_a_res = self.audio_adapter_blocks_p2[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
	
			f_v = f_v + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_v))))
			f_a = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_a))))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)

			if self.is_bg_cls:
				vis_patch =  f_v[:, 1:-(1+self.n_vis_tokens+self.n_shared_tokens), :]
				aud_patch = f_a[:, 1:-(1+self.n_audio_tokens+self.n_shared_tokens), :]
			else:
				vis_patch =  f_v[:, 1:-(self.n_vis_tokens+self.n_shared_tokens), :]
				aud_patch = f_a[:, 1:-(self.n_audio_tokens+self.n_shared_tokens), :]

			if self.contrastive == 'blockwise_sep':
				audio_feat = F.normalize(self.audio_proj[layer_count](aud_attn @ aud_patch).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj[layer_count](vis_attn @ vis_patch).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'blockwise_cmn':
				audio_feat = F.normalize(self.audio_proj(aud_attn @ aud_patch).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj(vis_attn @ vis_patch).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'final' and i == (self.total_layers-1):
				audio_feat = F.normalize(self.audio_proj(aud_attn @ aud_patch).squeeze(), dim=1).float()
				image_feat = F.normalize(self.vis_proj(vis_attn @ vis_patch).squeeze(), dim=1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			if self.layerwise_token is not None and i != (self.total_layers-1):
				f_a = f_a[:, :-(self.n_vis_tokens+self.n_shared_tokens) , :]
				f_v = f_v[:, :-(self.n_audio_tokens+self.n_shared_tokens), :]

			layer_count += 1

		if self.contrastive in ["blockwise_sep", "blockwise_cmn"]:
			cnt_loss = cnt_loss/self.total_layers

		f_v = self.ViT.v.norm(f_v)
		f_a = self.ViT.v.norm(f_a)

		v_cls = f_v[:, 0:1].squeeze()
		a_cls = f_a[:, 0:1].squeeze()

		fg_feat = torch.cat((v_cls, a_cls), dim=-1)
		p_fg = self.fg_class(fg_feat)

		if self.is_bg_cls:
			v_bg_cls = f_v[:, -(1+self.n_vis_tokens+self.n_shared_tokens):-(self.n_vis_tokens+self.n_shared_tokens)].squeeze()
			a_bg_cls = f_a[:, -(1+self.n_audio_tokens+self.n_shared_tokens):-(self.n_audio_tokens+self.n_shared_tokens)].squeeze()
			bg_feat = torch.cat((v_bg_cls, a_bg_cls), dim=-1)
			p_bg = self.bg_class(bg_feat).squeeze()
			
			bg_loss = self.calc_bg_loss(p_bg, bg_label)

		fg_loss = self.calc_fg_loss(p_fg, label, bg_label)

		loss = fg_loss + bg_loss + cnt_loss

		if self.grad_mod:
			coeff_v, coeff_a = self.calc_grad_coeff(v_cls, a_cls, label, bg_label)
		else:
			coeff_v, coeff_a = None, None

		return {
			'loss': loss,
			'fg_loss': fg_loss,
			'bg_loss': bg_loss if self.is_bg_cls else None,
			'cnt_loss': cnt_loss,
			'p_fg': p_fg,
			'p_bg': p_bg if self.is_bg_cls else None,
			'vis_attn': vis_attn,
			'coeff_a': coeff_a,
			'coeff_v': coeff_v
		}

	def forward(self, audio, vis, return_attn=False):
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, return_attn)

class MIM_Proj_AVIT(nn.Module):
	def __init__(self, opt):
		super(MIM_Proj_AVIT, self).__init__()
		
		assert opt.vis_encoder_type == 'vit', "only vit is supported for now"

		self.opt = opt
		self.ngpu = opt.ngpu
		self.rank = opt.rank

		if opt.vis_encoder_type == 'vit':
			if opt.vit_type == "tiny":
				self.ViT = my_vit('vit_tiny_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "small":
				self.ViT = my_vit(name='vit_small_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "base":
				self.ViT = my_vit(name='vit_base_patch16_224_in21k', pretrained=opt.pretrained)
			elif opt.vit_type == "large":
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)
			else:
				print("Nothing found. TinyViT is loading")
				self.ViT = my_vit(name='vit_large_patch16_224_in21k', pretrained=opt.pretrained)

			self.dim = self.ViT.v.patch_embed.proj.out_channels

			self.fg_class = nn.Linear(self.dim * 2, opt.num_class)

		hidden_list = []
		self.total_layers = len(self.ViT.v.blocks)

		self.contrastive = opt.contrastive
		self.LSA = opt.LSA

		if opt.contrastive in ['final' or 'blockwise_cmn']:
			self.audio_proj = nn.Linear(self.dim, 512)
			self.vis_proj = nn.Linear(self.dim, 512)
			self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

		elif opt.contrastive == 'blockwise_sep':
			self.audio_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 

			self.vis_proj = nn.ModuleList([
				nn.Linear(self.dim, 512)
			for i in range(self.total_layers)]) 
			
			self.logit_scale = nn.ParameterList([
				nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
			for i in range(self.total_layers)])

		if opt.vis_encoder_type == 'vit':
			for idx_layer, my_blk in enumerate(self.ViT.v.blocks) :
				hidden_d_size = my_blk.mlp.fc1.in_features
				hidden_list.append(hidden_d_size)

				my_blk.attn.forward = types.MethodType(attn_new_forward, my_blk.attn)

		self.lavish_adapter = opt.lavish_adapter

		if opt.lavish_adapter:
			self.audio_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck",dim_list=hidden_list, layer_idx=i,reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p1 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

			self.audio_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=self.opt.is_gate)
				for i in range(len(hidden_list))])

			self.vis_adapter_blocks_p2 = nn.ModuleList([
				VisualAdapter(input_dim=hidden_list[i], output_dim=hidden_list[i], adapter_kind="bottleneck", dim_list=hidden_list, layer_idx=i, reduction_factor=self.opt.Adapter_downsample, opt=opt, use_bn=self.opt.is_bn, use_gate=True)
				for i in range(len(hidden_list))])

		self.unimodal_token = opt.unimodal_token
		self.multimodal_token = opt.multimodal_token
		self.layerwise_token = opt.layerwise_token # "None" or "common" or "unique"
		self.layerwise_attn = opt.layerwise_attn # "None" or "common" or "unique"

		if opt.unimodal_token:
			self.audio_tokens = nn.Parameter(torch.randn(opt.n_audio_tokens, hidden_d_size))
			self.visual_tokens = nn.Parameter(torch.randn(opt.n_vis_tokens, hidden_d_size))
		if opt.multimodal_token:
			self.shared_tokens = nn.Parameter(torch.randn(opt.n_shared_tokens, hidden_d_size))

		self.n_audio_tokens = opt.n_audio_tokens
		self.n_vis_tokens = opt.n_vis_tokens
		self.n_shared_tokens = opt.n_shared_tokens

		self.audio_attn = nn.Identity()
		self.visual_attn = nn.Identity()
		self.shared_attn = nn.Identity()

		if opt.unimodal_token and opt.LSA:
			self.audio_attn = SelfAttention(hidden_d_size)
			self.visual_attn = SelfAttention(hidden_d_size)
		if opt.multimodal_token and opt.LSA:
			self.shared_attn = SelfAttention(hidden_d_size)

		self.is_bg_cls = opt.bg_cls
		if opt.bg_cls:
			self.bg_cls_token = nn.Parameter(torch.randn(1, hidden_d_size))
			self.bg_class = nn.Linear(self.dim * 2, 1)
			self.bg_label = opt.bg_label

		self.grad_mod = opt.grad_mod

	def calc_cnt_loss(self, image_feat, audio_feat, bg_label, layer_count):
		gather_image_feat = [torch.zeros_like(image_feat) for _ in range(self.ngpu)]
		gather_audio_feat = [torch.zeros_like(audio_feat) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			gather_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(gather_image_feat, image_feat)
		dist.all_gather(gather_audio_feat, audio_feat)
		if self.is_bg_cls:
			dist.all_gather(gather_bg_label, bg_label)
		
		gather_image_feat[self.rank] = image_feat
		gather_audio_feat[self.rank] = audio_feat

		all_image_feat = torch.cat(gather_image_feat, dim=0)
		all_audio_feat = torch.cat(gather_audio_feat, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(gather_bg_label, dim=0)

		if self.is_bg_cls:
			mask = torch.where(all_bg_label == 0)[0]
			all_image_feat = all_image_feat[mask]
			all_audio_feat = all_audio_feat[mask]
		
		labels = torch.arange(len(all_image_feat)).to(image_feat.device)

		Slogits = torch.einsum('ntd, md->nmt', all_image_feat, all_audio_feat) / 0.07
		logits = Slogits.max(dim=-1)[0]
		loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.permute(1, 0), labels)

		return loss


	def calc_grad_coeff(self, v_cls, a_cls, label, bg_label):
		'''
		return coeff_v, coeff_a
		'''
		weight_size = self.fg_class.weight.size(1)
		out_v = (torch.mm(v_cls, torch.transpose(self.fg_class.weight[:, weight_size // 2:], 0, 1))
					+ self.fg_class.bias / 2)

		out_a = (torch.mm(a_cls, torch.transpose(self.fg_class.weight[:, :weight_size // 2], 0, 1))
					+ self.fg_class.bias / 2)

		# Modulation starts here !
		all_out_v = [torch.zeros_like(out_v) for _ in range(self.ngpu)]
		all_out_a = [torch.zeros_like(out_a) for _ in range(self.ngpu)]
		all_label = [torch.zeros_like(label) for _ in range(self.ngpu)]
		if self.is_bg_cls:
			all_bg_label = [torch.zeros_like(bg_label) for _ in range(self.ngpu)]

		dist.all_gather(all_out_v, out_v)
		dist.all_gather(all_out_a, out_a)
		dist.all_gather(all_label, label)
		if self.is_bg_cls:
			dist.all_gather(all_bg_label, bg_label)

		all_out_v = torch.cat(all_out_v, dim=0)
		all_out_a = torch.cat(all_out_a, dim=0)
		all_label = torch.cat(all_label, dim=0)
		if self.is_bg_cls:
			all_bg_label = torch.cat(all_bg_label, dim=0)

			mask = torch.where(all_bg_label == 0)[0]
			all_out_v = all_out_v[mask]
			all_out_a = all_out_a[mask]
			all_label = all_label[mask]

		score_v = sum([F.softmax(all_out_v)[i][all_label[i]] for i in range(all_out_v.size(0))])
		score_a = sum([F.softmax(all_out_a)[i][all_label[i]] for i in range(all_out_a.size(0))])

		ratio_v = score_v / score_a
		ratio_a = 1 / ratio_v

		if ratio_v > 1:
			coeff_v = 1 - F.tanh(0.1 * F.relu(ratio_v))
			coeff_a = 1
		else:
			coeff_a = 1 - F.tanh(0.1 * F.relu(ratio_a))
			coeff_v = 1

		return coeff_v, coeff_a

	def calc_fg_loss(self, p_fg, label, bg_label):
		if self.is_bg_cls:
			mask = torch.where(bg_label == 0)[0]

			label = label[mask]
			p_fg = p_fg[mask]

		loss = F.cross_entropy(p_fg, label)

		return loss

	def calc_bg_loss(self, p_bg, bg_label):
		loss = F.binary_cross_entropy_with_logits(p_bg, bg_label.float())
		return loss 

	def forward_vit(self, audio, vis, label, return_attn=True):
		b, t, c, w, h = vis.shape

		label = label.view(-1)

		if self.is_bg_cls:
			bg_label = (label == self.bg_label).long()
		else:
			bg_label = None

		audio = repeat(audio, 'b t len dim -> b t c len dim', c=3)

		if self.is_bg_cls:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True, bg_cls_token=self.bg_cls_token)
		else:
			f_a, patch_info_audio = self.ViT.forward_patch(rearrange(audio, 'b t c w h -> (b t) c w h'), is_shape_info=True)
			f_v, patch_info_vis = self.ViT.forward_patch(rearrange(vis, 'b t c w h -> (b t) c w h'), is_shape_info=True)

		bs, dim, wa, ha = patch_info_audio
		bs, dim, wv, wv = patch_info_vis

		layer_count = 0
		cnt_loss = 0
		fg_loss = 0
		bg_loss = 0

		for i, blk in enumerate(self.ViT.v.blocks):
			if i == 0 or self.layerwise_token in ["unique", "common"]:
				if self.unimodal_token:
					audio_tokens = repeat(self.audio_tokens, 'len dim -> b len dim', b=f_a.shape[0])
					visual_tokens = repeat(self.visual_tokens, 'len dim -> b len dim', b=f_v.shape[0])

					audio_tokens = self.audio_attn(audio_tokens)
					visual_tokens = self.visual_attn(visual_tokens)

					f_a = torch.cat([
						f_a,
						audio_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						visual_tokens
					], dim=1)

				if self.multimodal_token:
					shared_tokens = repeat(self.shared_tokens, 'len dim -> b len dim', b=f_v.shape[0])
					shared_tokens = self.shared_attn(shared_tokens)
			
					f_a = torch.cat([
						f_a,
						shared_tokens
					], dim=1)

					f_v = torch.cat([
						f_v,
						shared_tokens
					], dim=1)

			if self.lavish_adapter:
				f_a_res = self.audio_adapter_blocks_p1[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p1[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))

			x, attn = blk.attn(blk.norm1(f_v), return_attn=True)
			
			if self.is_bg_cls:
				vis_attn = attn[:, :, 0, 1:-(1+self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)
			else:
				vis_attn = attn[:, :, 0, 1:-(self.n_vis_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				vis_attn = F.softmax(vis_attn, dim=-1)

			f_v = f_v + blk.drop_path1(blk.ls1(x))
		
			x, attn = blk.attn(blk.norm1(f_a), return_attn=True)
			
			if self.is_bg_cls:
				aud_attn = attn[:, :, 0, 1:-(1+self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)
			else:
				aud_attn = attn[:, :, 0, 1:-(self.n_audio_tokens+self.n_shared_tokens)].mean(dim=1, keepdim=True)
				aud_attn = F.softmax(aud_attn, dim=-1)

			f_a = f_a + blk.drop_path1(blk.ls1(x))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)
			
				f_a_res = self.audio_adapter_blocks_p2[i](f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
				f_v_res = self.vis_adapter_blocks_p2[i](f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1), f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :].permute(0,2,1).unsqueeze(-1))
	
			f_v = f_v + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_v))))
			f_a = f_a + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(f_a))))

			if self.lavish_adapter:
				f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] = f_v[:, :-(self.n_vis_tokens+self.n_shared_tokens), :] + f_v_res.squeeze(-1).permute(0,2,1)			
				f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] = f_a[:, :-(self.n_audio_tokens+self.n_shared_tokens), :] + f_a_res.squeeze(-1).permute(0,2,1)

			if self.is_bg_cls:
				vis_patch =  f_v[:, 1:-(1+self.n_vis_tokens+self.n_shared_tokens), :]
				aud_token = f_a[:, 0, :]
			else:
				vis_patch =  f_v[:, 1:-(self.n_vis_tokens+self.n_shared_tokens), :]
				aud_token = f_a[:, 0, :]

			if self.contrastive == 'blockwise_sep':
				audio_feat = F.normalize(self.audio_proj[layer_count](aud_token).squeeze(), dim=-1).float()
				image_feat = F.normalize(self.vis_proj[layer_count](vis_patch).squeeze(), dim=-1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'blockwise_cmn':
				audio_feat = F.normalize(self.audio_proj(aud_token).squeeze(), dim=-1).float()
				image_feat = F.normalize(self.vis_proj(vis_patch).squeeze(), dim=-1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			elif self.contrastive == 'final' and i == (self.total_layers-1):
				audio_feat = F.normalize(self.audio_proj(aud_token).squeeze(), dim=-1).float()
				image_feat = F.normalize(self.vis_proj(vis_patch).squeeze(), dim=-1).float()
				cnt_loss += self.calc_cnt_loss(image_feat, audio_feat, bg_label, layer_count)

			if self.layerwise_token is not None and i != (self.total_layers-1):
				f_a = f_a[:, :-(self.n_vis_tokens+self.n_shared_tokens) , :]
				f_v = f_v[:, :-(self.n_audio_tokens+self.n_shared_tokens), :]

			layer_count += 1

		if self.contrastive in ["blockwise_sep", "blockwise_cmn"]:
			cnt_loss = cnt_loss/self.total_layers

		f_v = self.ViT.v.norm(f_v)
		f_a = self.ViT.v.norm(f_a)

		v_cls = f_v[:, 0:1].squeeze()
		a_cls = f_a[:, 0:1].squeeze()

		fg_feat = torch.cat((v_cls, a_cls), dim=-1)
		p_fg = self.fg_class(fg_feat)

		if self.is_bg_cls:
			v_bg_cls = f_v[:, -(1+self.n_vis_tokens+self.n_shared_tokens):-(self.n_vis_tokens+self.n_shared_tokens)].squeeze()
			a_bg_cls = f_a[:, -(1+self.n_audio_tokens+self.n_shared_tokens):-(self.n_audio_tokens+self.n_shared_tokens)].squeeze()
			bg_feat = torch.cat((v_bg_cls, a_bg_cls), dim=-1)
			p_bg = self.bg_class(bg_feat).squeeze()
			
			bg_loss = self.calc_bg_loss(p_bg, bg_label)

		fg_loss = self.calc_fg_loss(p_fg, label, bg_label)

		loss = fg_loss + bg_loss + cnt_loss

		if self.grad_mod:
			coeff_v, coeff_a = self.calc_grad_coeff(v_cls, a_cls, label, bg_label)
		else:
			coeff_v, coeff_a = None, None

		return {
			'loss': loss,
			'fg_loss': fg_loss,
			'bg_loss': bg_loss if self.is_bg_cls else None,
			'cnt_loss': cnt_loss,
			'p_fg': p_fg,
			'p_bg': p_bg if self.is_bg_cls else None,
			'vis_attn': vis_attn,
			'coeff_a': coeff_a,
			'coeff_v': coeff_v
		}

	def forward(self, audio, vis, return_attn=False):
		if self.opt.vis_encoder_type == 'vit':
			return self.forward_vit(audio, vis, return_attn)