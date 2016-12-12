package preprocessing;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.HashSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import preprocessing.RatingsPreprocessor.RatingsPreprocessorReducer;

public class IdDictionaryGenerator {
	private final static int LINES_PER_REVIEW = 11;		//If only MultiLineInputFormat split on multiples of this.
	private final static int REVIEWS_PER_BLOCK = 60000;	//Instead, we do this so we don't get 400,000,000 mappers.
	private final static int BLOCKS_PER_SPLIT = 3;		//This gives us a reasonable 3x64mb chunk per mapper.
	private final static int LINES_PER_SPLIT = (LINES_PER_REVIEW * REVIEWS_PER_BLOCK * BLOCKS_PER_SPLIT);
	private final static int MIN_REVIEWS = 2;	//minimum number of reviews a product must have to be recorded
	
	/**
	 * Input:	offset				reviews
	 * Output:	productId			1
	 */
	public static class IdDictMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
		ArrayDeque<String> valTerms = new ArrayDeque<String>();
		
		@Override
		protected void setup(Mapper<LongWritable, Text, Text, IntWritable>.Context context)
				throws IOException, InterruptedException {			
			
			super.setup(context);
			
			valTerms.add("product/productId: ");
//			valTerms.add("review/userId: ");	
		}

		@Override
		public void map(LongWritable offset, Text text, Context context) throws IOException, InterruptedException {
			IntWritable one = new IntWritable(1);
			ArrayDeque<String> reviews = new ArrayDeque<String>();
			Collections.addAll(reviews, text.toString().split("review/text: .*\n\n"));	
			
			for(String review : reviews){	
				if(review.contains("review/userId: unknown"))	//skip
					continue;
				
				for(String valTerm : valTerms){
					int vStart = review.indexOf(valTerm) + valTerm.length();
					int vEnd = review.indexOf("\n", vStart);
					String val = review.substring(vStart, vEnd);
					context.write(new Text(val), one);
				}				
			}
		}
	}
	
	/**
	 * Input:	productId			[1,1,...,1]
	 * Output:	productId			sum
	 */
	public static class IdDictCombiner extends Reducer<Text, IntWritable, Text, IntWritable>{		
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int sum = 0;
			for(IntWritable val : values){
				sum += val.get();
			}
			context.write(key, new IntWritable(sum));			
		}
	}
	
	/**
	 * Input:	productId			sums
	 * Output:	Null				hash(productId),productId
	 */
	public static class IdDictReducer extends Reducer<Text, IntWritable, NullWritable, Text>{		
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int count = 0;
			NullWritable nothing = NullWritable.get();
			
			for(IntWritable value : values){
				count += value.get();
			}
			
			if(count >= MIN_REVIEWS){
				context.write(nothing, new Text(String.valueOf(key.toString().hashCode()) + "," + key));
			}				
		}
	}

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException, URISyntaxException {
		Configuration conf = new Configuration();
		conf.set("mapred.job.queue.name", "hadoop02");
		
		GenericOptionsParser gop = new GenericOptionsParser(conf, args);
		String[] otherArgs = gop.getRemainingArgs();
		Job job = Job.getInstance(conf, "id dictionary generator");
		
		job.setJarByClass(RatingsPreprocessor.class);
		job.setMapperClass(IdDictMapper.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setInputFormatClass(MultiLineInputFormat.class);
		MultiLineInputFormat.setInputPaths(job, new Path(otherArgs[0]));
		MultiLineInputFormat.setNumLinesPerSplit(job, LINES_PER_SPLIT);	
		job.setCombinerClass(IdDictCombiner.class);
		job.setReducerClass(IdDictReducer.class);
		job.setOutputKeyClass(NullWritable.class);
		job.setOutputValueClass(Text.class);
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		try{
			System.exit(job.waitForCompletion(true) ? 0 : 1);
		}
		catch (ClassNotFoundException e){
			e.printStackTrace();
		}
		catch (InterruptedException e){
			e.printStackTrace();
		}
	}
}